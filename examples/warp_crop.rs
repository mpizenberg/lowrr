use lowrr::img::crop::{crop, Crop};
use lowrr::img::registration;
use lowrr::interop;

use glob::glob;
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView};
use nalgebra::{DMatrix, Vector6};
use std::io::prelude::*;
use std::path::PathBuf;

// Default values for some of the program arguments.
const DEFAULT_OUT_DIR: &str = "generated";
const DEFAULT_FLOW: f64 = 0.01;

/// Entry point of the program.
fn main() {
    parse_args()
        .and_then(run)
        .unwrap_or_else(|err| eprintln!("Error: {:?}", err));
}

fn display_help() {
    eprintln!(
        r#"
warp_crop

Generate a random translation of 1% and crop the defined crop area.
It will generate images in generated/cropped/.
First image should be warped with identity to keep the same reference frame.
Save the warp transformation of every image, in the frame of the cropped area,
in a file called generated/warp-gt.txt.

USAGE: warp_crop [FLAGS...] IMAGE_FILES...

    warp_crop --flow 0.01 --crop x1,y1,x2,y2 --out-dir generated path/to/images/*.png

FLAGS:
    --help                 # Print this message and exit
    --flow                 # Max random optical flow, in percent of image size (default: {})
    --crop x1,y1,x2,y2     # Crop image into a restricted working area (use no space between coordinates)
    --out-dir dir/         # Output directory to save registered images (default: {})
"#,
        DEFAULT_OUT_DIR, DEFAULT_FLOW,
    )
}

#[derive(Debug)]
/// Type holding command line arguments.
struct Args {
    help: bool,
    flow: f64,
    out_dir: String,
    images_paths: Vec<PathBuf>,
    crop: Option<Crop>,
}

/// Function parsing the command line arguments and returning an Args object or an error.
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = pico_args::Arguments::from_env();

    // Retrieve command line arguments.
    let help = args.contains(["-h", "--help"]);
    let out_dir = args
        .opt_value_from_str("--out-dir")?
        .unwrap_or(DEFAULT_OUT_DIR.into());
    let flow = args
        .opt_value_from_str("--flow")?
        .unwrap_or(DEFAULT_FLOW.into());
    let crop = args.opt_value_from_str("--crop")?;

    // Verify that images paths are correct.
    let free_args = args.free()?;
    let images_paths = absolute_file_paths(&free_args)?;

    // Return Args struct.
    Ok(Args {
        help,
        flow,
        out_dir,
        images_paths,
        crop,
    })
}

/// Retrieve the absolute paths of all files matching the arguments.
fn absolute_file_paths(args: &[String]) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut abs_paths = Vec::new();
    for path_glob in args {
        let mut paths = paths_from_glob(path_glob)?;
        abs_paths.append(&mut paths);
    }
    abs_paths
        .iter()
        .map(|p| p.canonicalize().map_err(|e| e.into()))
        .collect()
}

/// Retrieve the paths of files matching the glob pattern.
fn paths_from_glob(p: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let paths = glob(p)?;
    Ok(paths.into_iter().filter_map(|x| x.ok()).collect())
}

/// Start actual program with command line arguments successfully parsed.
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check if the --help flag is present.
    if args.help {
        display_help();
        std::process::exit(0);
    }

    // Get the path of output directory.
    let out_dir_path = PathBuf::from(args.out_dir);
    let warp_dir = out_dir_path.join("cropped");
    let warp_txt = warp_dir.join("warp-gt.txt");
    std::fs::create_dir_all(&warp_dir)
        .expect(&format!("Could not create output dir: {:?}", &warp_dir));
    let mut warp_txt_file = std::fs::File::create(&warp_txt)?;

    // Display progress bar.
    let img_count = args.images_paths.len();
    let pb = indicatif::ProgressBar::new(img_count as u64);

    // Use the time as a random generator.
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32;

    // Warp images.
    for (id, img_path) in args.images_paths.iter().enumerate() {
        // Read the image.
        let dyn_img = ImageReader::open(img_path)?.decode()?;
        let (width, height) = dyn_img.dimensions();

        // Create a random transformation. https://stackoverflow.com/a/3062783
        let r1 = (1103515245 * seed + 12345) % 2147483648;
        let r2 = (1103515245 * r1 + 12345) % 2147483648;
        seed = r2;
        let min_size = width.min(height) as f64;
        // 5% of size translations max.
        let translation_scale = args.flow * min_size;
        // Values seems to be in [0.0, 0.5[ so I multiply them by 2.
        let tx = (r1 as f64 / (u32::MAX as f64) * 2.0 - 0.5) * translation_scale;
        let ty = (r2 as f64 / (u32::MAX as f64) * 2.0 - 0.5) * translation_scale;

        // Do not warp the first image.
        let motion = if id == 0 {
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            Vector6::new(0.0, 0.0, 0.0, 0.0, tx as f32, ty as f32)
        };

        // TODO: update motion to the cropped area.
        // but for now it's ok since it's just a translation.
        warp_txt_file.write_all(
            format!(
                "{}, {}, {}, {}, {}, {}\n",
                motion[0], motion[1], motion[2], motion[3], motion[4], motion[5]
            )
            .as_bytes(),
        )?;

        match dyn_img {
            DynamicImage::ImageRgb16(img) => {
                let img_mat = interop::matrix_from_rgb_image(img);

                // Warp the image.
                let warp_img_mat: DMatrix<(u16, u16, u16)> = registration::warp(&img_mat, &motion);

                // Crop it to the provided area.
                let cropped_img = match &args.crop {
                    None => warp_img_mat,
                    Some(frame) => crop(frame, &warp_img_mat),
                };

                // Only keep one channel (green).
                let cropped_img = cropped_img.map(|(_r, g, _b)| g);

                // Equalize mean intensities of cropped area.
                let mut temp = vec![cropped_img];
                lowrr::utils::equalize_mean(0.15, temp.as_mut_slice());
                let cropped_img = temp.pop().unwrap();

                // Save the image to disk.
                let warp_img = interop::image_from_matrix(&cropped_img);
                warp_img.save(warp_dir.join(format!("{:02}.png", id)))?;
            }
            _ => todo!(),
        }
        pb.inc(1);
    }
    pb.finish();
    Ok(())
}
