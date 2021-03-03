use lowrr::img::crop::{crop, recover_original_motion, Crop};
use lowrr::img::registration;
use lowrr::interop::IntoDMatrix;

use glob::glob;
use image::DynamicImage;
use nalgebra::{DMatrix, Scalar};
use std::path::{Path, PathBuf};

// Default values for some of the program arguments.
const DEFAULT_OUT_DIR: &str = "out";
const DEFAULT_LEVELS: usize = 1;
const DEFAULT_LAMBDA: f32 = 1.5;
const DEFAULT_RHO: f32 = 0.1;
const DEFAULT_THRESHOLD: f32 = 1e-3;
const DEFAULT_SPARSE_RATIO_THRESHOLD: f32 = 0.5;
const DEFAULT_MAX_ITERATIONS: usize = 40;
const DEFAULT_IMAGE_MAX: f32 = 255.0;

/// Entry point of the program.
fn main() {
    parse_args()
        .and_then(run)
        .unwrap_or_else(|err| eprintln!("Error: {:?}", err));
}

fn display_help() {
    eprintln!(
        r#"
lowrr

Low-rank registration of slightly unaligned images for photometric stereo.
Some algorithm info is output to stderr while running.
You can ignore them by redirecting stderr to /dev/null.
The final motion vector is written to stdout,
you can redirect it to a file with the usual pipes.

USAGE:
    lowrr [FLAGS] IMAGE_FILES
    For example:
        lowrr --trace *.png
        lowrr *.jpg 2> /dev/null
        lowrr *.png > result.txt

FLAGS:
    --help                 # Print this message and exit
    --version              # Print version and exit
    --out-dir dir/         # Output directory to save registered images (default: {})
    --trace                # Print more debug output to stderr while running
    --crop x1,y1,x2,y2     # Crop image into a restricted working area (use no space between coordinates)
    --no-image-correction  # Avoid image correction
    --levels int           # Number of levels for the multi-resolution approach (default: {})
    --lambda float         # Weight of the L1 term (high means no correction) (default: {})
    --rho float            # Lagrangian penalty (default: {})
    --threshold float      # Stop when relative diff between two estimate of corrected image falls below this (default: {})
    --sparse float         # Sparse ratio threshold to switch between dense and sparse resolution (default: {})
                           # Use dense resolution if the ratio at current level is higher than this threshold
    --max-iterations int   # Maximum number of iterations (default: {})
    --image-max float      # Maximum possible value of the images for scaling (default: {})
"#,
        DEFAULT_OUT_DIR,
        DEFAULT_LEVELS,
        DEFAULT_LAMBDA,
        DEFAULT_RHO,
        DEFAULT_THRESHOLD,
        DEFAULT_SPARSE_RATIO_THRESHOLD,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_IMAGE_MAX,
    )
}

#[derive(Debug)]
/// Type holding command line arguments.
struct Args {
    config: registration::Config,
    help: bool,
    version: bool,
    out_dir: String,
    images_paths: Vec<PathBuf>,
    crop: Option<Crop>,
}

/// Function parsing the command line arguments and returning an Args object or an error.
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = pico_args::Arguments::from_env();

    // Retrieve command line arguments.
    let help = args.contains(["-h", "--help"]);
    let version = args.contains(["-v", "--version"]);
    let do_image_correction = !args.contains("--no-image-correction");
    let trace = args.contains("--trace");
    let crop = args.opt_value_from_str("--crop")?;
    let lambda = args
        .opt_value_from_str("--lambda")?
        .unwrap_or(DEFAULT_LAMBDA);
    let rho = args.opt_value_from_str("--rho")?.unwrap_or(DEFAULT_RHO);
    let threshold = args
        .opt_value_from_str("--threshold")?
        .unwrap_or(DEFAULT_THRESHOLD);
    let sparse_ratio_threshold = args
        .opt_value_from_str("--sparse")?
        .unwrap_or(DEFAULT_SPARSE_RATIO_THRESHOLD);
    let max_iterations = args
        .opt_value_from_str("--max-iterations")?
        .unwrap_or(DEFAULT_MAX_ITERATIONS);
    let levels = args
        .opt_value_from_str("--levels")?
        .unwrap_or(DEFAULT_LEVELS);
    let image_max = args
        .opt_value_from_str("--image-max")?
        .unwrap_or(DEFAULT_IMAGE_MAX);
    let out_dir = args
        .opt_value_from_str("--out-dir")?
        .unwrap_or(DEFAULT_OUT_DIR.into());

    // Verify that images paths are correct.
    let free_args = args.free()?;
    let images_paths = absolute_file_paths(&free_args)?;

    // Return Args struct.
    Ok(Args {
        config: registration::Config {
            do_image_correction,
            trace,
            lambda,
            rho,
            threshold,
            sparse_ratio_threshold,
            max_iterations,
            levels,
            image_max,
        },
        help,
        version,
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

/// Retrieve the paths of files matchin the glob pattern.
fn paths_from_glob(p: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let paths = glob(p)?;
    Ok(paths.into_iter().filter_map(|x| x.ok()).collect())
}

/// Start actual program with command line arguments successfully parsed.
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check if the --help or --version flags are present.
    if args.help {
        display_help();
        std::process::exit(0);
    } else if args.version {
        println!("{}", std::env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }

    // Get the path of output directory.
    let out_dir_path = PathBuf::from(args.out_dir);

    // Load the dataset in memory.
    let now = std::time::Instant::now();
    let (dataset, _) = load_dataset(&args.images_paths)?;
    eprintln!("Loading took {:.1} s", now.elapsed().as_secs_f32());
    // panic!("stop");

    // Use the algorithm corresponding to the type of data.
    match dataset {
        Dataset::GrayImages(_) => unimplemented!(),
        Dataset::RgbImages(imgs) => {
            let now = std::time::Instant::now();

            // Convert RGB into gray
            let gray_imgs: Vec<_> = imgs.iter().map(|im| im.map(|(_r, g, _b)| g)).collect();

            // Extract the cropped area from the images.
            let mut cropped_imgs: Vec<_> = match &args.crop {
                None => gray_imgs,
                Some(frame) => gray_imgs.iter().map(|im| crop(frame, im)).collect(),
            };

            // Equalize mean intensities of cropped area.
            lowrr::utils::equalize_mean(0.15, &mut cropped_imgs);

            // Compute the motion of each image for registration.
            let (motion_vec_crop, cropped_imgs) =
                registration::gray_affine(args.config, cropped_imgs, 50)?;
            let motion_vec = match &args.crop {
                None => motion_vec_crop.clone(),
                Some(frame) => recover_original_motion(frame, &motion_vec_crop),
            };
            eprintln!("Registration took {:.1} s", now.elapsed().as_secs_f32());

            // // Reproject (interpolation + extrapolation) images according to that motion.
            // // Write the registered images to the output directory.
            // eprintln!("Saving registered images");
            // let registered_imgs = registration::reproject(&imgs, &motion_vec);
            // drop(imgs);
            // lowrr::utils::save_rgbu8_imgs(&out_dir_path, &registered_imgs);
            // drop(registered_imgs);

            // Visualization of registered cropped images.
            eprintln!("Saving registered cropped images");
            let registered_cropped_imgs: Vec<DMatrix<u8>> =
                registration::reproject(&cropped_imgs, &motion_vec_crop);
            let cropped_aligned_dir = &out_dir_path.join("cropped_aligned");
            lowrr::utils::save_imgs(&cropped_aligned_dir, &registered_cropped_imgs);
            drop(registered_cropped_imgs);

            // Visualization of original cropped images.
            eprintln!("Saving original cropped images");
            let cropped_dir = &out_dir_path.join("cropped");
            lowrr::utils::save_imgs(&cropped_dir, &cropped_imgs);
            drop(cropped_imgs);

            // Write motion_vec to stdout.
            for v in motion_vec.iter() {
                println!("{}", v);
            }
            Ok(())
        }
        Dataset::RgbImagesU16(imgs) => {
            let now = std::time::Instant::now();

            // Convert RGB into gray
            let gray_imgs: Vec<_> = imgs.iter().map(|im| im.map(|(_r, g, _b)| g)).collect();

            // Extract the cropped area from the images.
            let mut cropped_imgs = match &args.crop {
                None => gray_imgs,
                Some(frame) => gray_imgs.iter().map(|im| crop(frame, im)).collect(),
            };

            // Equalize mean intensities of cropped area.
            lowrr::utils::equalize_mean(0.15, &mut cropped_imgs);

            // Visualization of original cropped images.
            eprintln!("Saving original cropped images");
            let cropped_dir = &out_dir_path.join("cropped");
            lowrr::utils::save_imgs(&cropped_dir, &cropped_imgs);

            // Compute the motion of each image for registration.
            let (motion_vec_crop, cropped_imgs) =
                registration::gray_affine(args.config, cropped_imgs, 10 * 256)?;
            let motion_vec = match &args.crop {
                None => motion_vec_crop.clone(),
                Some(frame) => recover_original_motion(frame, &motion_vec_crop),
            };
            eprintln!("Registration took {:.1} s", now.elapsed().as_secs_f32());

            // // Reproject (interpolation + extrapolation) images according to that motion.
            // // Write the registered images to the output directory.
            // eprintln!("Saving registered images");
            // let registered_imgs = registration::reproject(&imgs, &motion_vec);
            // drop(imgs);
            // lowrr::utils::save_rgbu8_imgs(&out_dir_path, &registered_imgs);
            // drop(registered_imgs);

            // Visualization of registered cropped images.
            eprintln!("Saving registered cropped images");
            let registered_cropped_imgs: Vec<DMatrix<u16>> =
                registration::reproject(&cropped_imgs, &motion_vec_crop);
            let cropped_aligned_dir = &out_dir_path.join("cropped_aligned");
            lowrr::utils::save_imgs(&cropped_aligned_dir, &registered_cropped_imgs);
            drop(registered_cropped_imgs);

            // Write motion_vec to stdout.
            for v in motion_vec.iter() {
                println!("{}", v);
            }
            Ok(())
        }
        Dataset::RawImages(_) => unimplemented!(),
    }
}

enum Dataset {
    RawImages(Vec<DMatrix<u16>>),
    GrayImages(Vec<DMatrix<u8>>),
    RgbImages(Vec<DMatrix<(u8, u8, u8)>>),
    RgbImagesU16(Vec<DMatrix<(u16, u16, u16)>>),
}

/// Load all images into memory.
fn load_dataset<P: AsRef<Path>>(
    paths: &[P],
) -> Result<(Dataset, (usize, usize)), Box<dyn std::error::Error>> {
    eprintln!("Images to be processed:");
    let images_types: Vec<_> = paths
        .iter()
        .map(|path| {
            eprintln!("    {:?}", path.as_ref());
            match path
                .as_ref()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .as_deref()
            {
                Some("nef") => "raw",
                Some("png") => "image",
                Some("jpg") => "image",
                Some("jpeg") => "image",
                Some(ext) => panic!("Unrecognized extension: {}", ext),
                None => panic!("Hum no extension?"),
            }
        })
        .collect();

    if images_types.is_empty() {
        Err("There is no such image. Use --help to know how to use this tool.".into())
    } else if images_types.iter().all(|&t| t == "raw") {
        unimplemented!("imread raw")
    } else if images_types.iter().all(|&t| t == "image") {
        // Open the first image to figure out the image type.
        match image::open(&paths[0])? {
            DynamicImage::ImageRgb8(rgb_img_0) => {
                let (imgs, (height, width)) =
                    load_all(DynamicImage::ImageRgb8(rgb_img_0), &paths[1..]);
                Ok((Dataset::RgbImages(imgs), (width, height)))
            }
            DynamicImage::ImageRgb16(rgb_img_0) => {
                let (imgs, (height, width)) =
                    load_all(DynamicImage::ImageRgb16(rgb_img_0), &paths[1..]);
                Ok((Dataset::RgbImagesU16(imgs), (width, height)))
            }
            _ => Err("Unknow image type".into()),
        }
    } else {
        panic!("There is a mix of image types")
    }
}

fn load_all<P: AsRef<Path>, Pixel, T: Scalar>(
    first_img: DynamicImage,
    other_paths: &[P],
) -> (Vec<DMatrix<T>>, (usize, usize))
where
    DynamicImage: IntoDMatrix<Pixel, T>,
{
    let img_count = 1 + other_paths.len();
    eprintln!("Loading {} images ...", img_count);
    let pb = indicatif::ProgressBar::new(img_count as u64);
    let mut imgs = Vec::with_capacity(img_count);
    let img_mat = first_img.into_dmatrix();
    let shape = img_mat.shape();
    imgs.push(img_mat);
    pb.inc(1);
    for rgb_img in other_paths.iter().map(|p| image::open(p).unwrap()) {
        imgs.push(rgb_img.into_dmatrix());
        pb.inc(1);
    }
    pb.finish();
    (imgs, shape)
}
