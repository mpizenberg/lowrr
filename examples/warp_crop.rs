use lowrr::img::crop::{crop, Crop};
use lowrr::img::interpolation::CanLinearInterpolate;
use lowrr::img::registration;
use lowrr::img::viz::ToGray;
use lowrr::interop::{IntoDMatrix, IntoImage};
use lowrr::utils::CanEqualize;

use anyhow::Context;
use glob::glob;
use image::io::Reader as ImageReader;
use image::GenericImageView;
use nalgebra::{DMatrix, Scalar, Vector6};
use std::convert::TryFrom;
use std::io::prelude::*;
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};

// Default values for some of the program arguments.
const DEFAULT_MAX_DISPLACEMENT: &str = "0.01";
const DEFAULT_OUT_DIR: &str = "generated";

/// Entry point of the program.
fn main() -> anyhow::Result<()> {
    // CLI arguments.
    let args = vec![
        clap::Arg::with_name("equalize")
            .long("equalize")
            .value_name("x")
            .help("Equalize the mean intensity of all images. Value in [0.0, 1.0]."),
        clap::Arg::with_name("max-displacement")
            .long("max-displacement")
            .value_name("x")
            .default_value(DEFAULT_MAX_DISPLACEMENT)
            .help("Max displacement (in ratio with the image size)."),
        clap::Arg::with_name("crop")
            .long("crop")
            .number_of_values(4)
            .value_names(&["left", "top", "right", "bottom"])
            .use_delimiter(true)
            .help("Crop image into a restricted working area"),
        clap::Arg::with_name("out-dir")
            .long("out-dir")
            .default_value(DEFAULT_OUT_DIR)
            .value_name("path")
            .help("Output directory to save registered images"),
        clap::Arg::with_name("IMAGE or GLOB")
            .multiple(true)
            .required(true)
            .help("Paths to images, or glob pattern such as \"img/*.png\""),
    ];
    // Read all CLI arguments.
    let matches = clap::App::new("warp_crop")
        .version(std::env!("CARGO_PKG_VERSION"))
        .about(
            "
Generate random translations and crop images with the provided frame.

The first image is not warped to keep its reference frame.
The generated translations of each image are saved in a text file: warp-gt.txt.",
        )
        .args(&args)
        .get_matches();
    // Start program.
    run(get_args(&matches)?)
}

#[derive(Debug)]
/// Type holding command line arguments.
struct Args {
    equalize: Option<f32>,
    max_displacement: f32,
    out_dir: String,
    images_paths: Vec<PathBuf>,
    crop: Option<Crop>,
}

/// Retrieve the program arguments from clap matches.
fn get_args(matches: &clap::ArgMatches) -> anyhow::Result<Args> {
    // Retrieving the equalize argument.
    let equalize = match matches.value_of("equalize") {
        None => None,
        Some(str_value) => {
            let value = str_value
                .parse()
                .context(format!("Failed to parse \"{}\" into a float", str_value))?;
            if value < 0.0 || value > 1.0 {
                anyhow::bail!("Expecting an intensity value in [0,1], got {}", value)
            }
            Some(value)
        }
    };

    // Retrieving the crop argument.
    let crop = match matches.values_of("crop") {
        None => None,
        Some(str_coords) => Some(Crop::try_from(str_coords)?),
    };

    Ok(Args {
        equalize,
        max_displacement: matches.value_of("max-displacement").unwrap().parse()?,
        out_dir: matches.value_of("out-dir").unwrap().to_string(),
        images_paths: absolute_file_paths(matches.values_of("IMAGE or GLOB").unwrap())?,
        crop,
    })
}

/// Retrieve the absolute paths of all files matching the arguments.
fn absolute_file_paths<S: AsRef<str>, Paths: Iterator<Item = S>>(
    args: Paths,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut abs_paths = Vec::new();
    for path_glob in args {
        let mut paths = paths_from_glob(path_glob.as_ref())?;
        abs_paths.append(&mut paths);
    }
    abs_paths
        .iter()
        .map(|p| p.canonicalize().map_err(|e| e.into()))
        .collect()
}

/// Retrieve the paths of files matchin the glob pattern.
fn paths_from_glob(p: &str) -> anyhow::Result<Vec<PathBuf>> {
    let paths = glob(p)?;
    Ok(paths.into_iter().filter_map(|x| x.ok()).collect())
}

/// Start actual program with command line arguments successfully parsed.
fn run(args: Args) -> anyhow::Result<()> {
    // Get the path of output directory.
    let out_dir_path = PathBuf::from(args.out_dir);
    let warp_dir = out_dir_path.join("cropped");
    let warp_txt = warp_dir.join("warp-gt.txt");
    std::fs::create_dir_all(&warp_dir).context(format!(
        "Could not create output dir: {}",
        warp_dir.display()
    ))?;
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
        let min_size = width.min(height);
        let translation_scale = args.max_displacement * min_size as f32;
        // Values seems to be in [0.0, 0.5[ so I multiply them by 2.
        let tx = (r1 as f64 / (u32::MAX as f64) * 2.0 - 0.5) as f32 * translation_scale;
        let ty = (r2 as f64 / (u32::MAX as f64) * 2.0 - 0.5) as f32 * translation_scale;

        // Do not warp the first image.
        let motion = if id == 0 {
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            Vector6::new(0.0, 0.0, 0.0, 0.0, tx, ty)
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

        let save_path = warp_dir.join(format!("{:02}.png", id));
        if dyn_img.as_luma8().is_some() {
            warp_crop::<_, u8, _, u8, _>(dyn_img, motion, args.equalize, args.crop, &save_path)?;
        } else if dyn_img.as_luma16().is_some() {
            warp_crop::<_, u16, _, u16, _>(dyn_img, motion, args.equalize, args.crop, &save_path)?;
        } else if dyn_img.as_rgb8().is_some() {
            warp_crop::<_, (u8, u8, u8), _, (u8, u8, u8), _>(
                dyn_img,
                motion,
                args.equalize,
                args.crop,
                &save_path,
            )?;
        } else if dyn_img.as_rgb16().is_some() {
            warp_crop::<_, (u16, u16, u16), _, (u16, u16, u16), _>(
                dyn_img,
                motion,
                args.equalize,
                args.crop,
                &save_path,
            )?;
        } else {
            panic!("Unknown image type");
        }

        pb.inc(1);
    }
    pb.finish();
    Ok(())
}

fn warp_crop<P, T, V, O, I>(
    img: I,
    motion: Vector6<f32>,
    equalize: Option<f32>,
    crop_frame: Option<Crop>,
    save_path: &Path,
) -> anyhow::Result<()>
where
    O: Scalar + ToGray,
    <O as ToGray>::Output: CanEqualize,
    DMatrix<<O as ToGray>::Output>: IntoImage,
    V: Add<Output = V>,
    f32: Mul<V, Output = V>,
    T: Scalar + Copy + CanLinearInterpolate<V, O>,
    I: IntoDMatrix<P, T>,
{
    let mat = img.into_dmatrix();
    let warp_mat = registration::warp(&mat, &motion);

    // Crop it to the provided area.
    let cropped_mat = match crop_frame {
        None => warp_mat,
        Some(frame) => crop(frame, &warp_mat)?,
    };

    // Transform image to gray.
    let mut cropped_mat = cropped_mat.map(ToGray::to_gray);

    // Equalize mean intensities of cropped area.
    if let Some(target) = equalize {
        let mut temp = vec![cropped_mat];
        lowrr::utils::equalize_mean(target, temp.as_mut_slice());
        cropped_mat = temp.pop().unwrap();
    }

    // Save the image to disk.
    let warp_img = cropped_mat.into_image();
    warp_img.save(save_path)?;
    Ok(())
}
