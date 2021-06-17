// SPDX-License-Identifier: MPL-2.0

use lowrr::img::interpolation::CanLinearInterpolate;
use lowrr::img::registration;
use lowrr::interop::{IntoDMatrix, ToImage};

use anyhow::Context;
use glob::glob;
use image::io::Reader as ImageReader;
use nalgebra::{DMatrix, Scalar, Vector6};
use std::io::prelude::*;
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};

/// Entry point of the program.
fn main() -> anyhow::Result<()> {
    // CLI arguments.
    let args = vec![
        clap::Arg::with_name("WARP.csv")
            .required(true)
            .help("Path to the csv file containing the warp parameters."),
        clap::Arg::with_name("IMAGE or GLOB")
            .multiple(true)
            .required(true)
            .help("Paths to images, or glob pattern such as \"img/*.png\""),
    ];
    // Read all CLI arguments.
    let matches = clap::App::new("black_extrap")
        .version(std::env!("CARGO_PKG_VERSION"))
        .about(
            "
Transform warped images by replacing extrapolated areas with black pixels.

This reads the csv file containing the affine warp parameters for each image.
Then for each associated image, it creates a copy where the extrapolated pixels
(outside of the original image) are replaced by black pixels.",
        )
        .args(&args)
        .get_matches();
    // Start program.
    run(get_args(&matches)?)
}

#[derive(Debug)]
/// Type holding command line arguments.
struct Args {
    warp_file: PathBuf,
    images_paths: Vec<PathBuf>,
}

/// Retrieve the program arguments from clap matches.
fn get_args(matches: &clap::ArgMatches) -> anyhow::Result<Args> {
    let warp_path = Path::new(matches.value_of("WARP.csv").unwrap()).canonicalize()?;
    Ok(Args {
        warp_file: warp_path,
        images_paths: absolute_file_paths(matches.values_of("IMAGE or GLOB").unwrap())?,
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
    // Read warp file.
    let file = std::fs::File::open(&args.warp_file)?;
    let mut warps = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let warp_line =
            Vector6::from_iterator(line?.split(", ").filter_map(|x| x.parse::<f32>().ok()));
        warps.push(warp_line);
    }

    // Display progress bar.
    let img_count = warps.len();
    let pb = indicatif::ProgressBar::new(img_count as u64);

    // Read the associated image files.
    // By convention, they are named 0.png to N.png where N is the number of images.
    // For each image, apply the black extrapolation change
    // and save the image inside the directory "black-extrap/".
    let black_warp_dir = args.warp_file.parent().unwrap().join("black_extrap");
    std::fs::create_dir_all(&black_warp_dir).context(format!(
        "Could not create output dir: {}",
        black_warp_dir.display()
    ))?;
    for (id, img_path) in args.images_paths.iter().enumerate() {
        // Read the image.
        let dyn_img = ImageReader::open(img_path)?.decode()?;
        let motion = warps[id];
        let save_path = black_warp_dir.join(format!("{}.png", id));

        if dyn_img.as_luma8().is_some() {
            warp_black_extrap::<_, u8, _, u8, _>(dyn_img, motion, &save_path)?;
        } else if dyn_img.as_luma16().is_some() {
            warp_black_extrap::<_, u16, _, u16, _>(dyn_img, motion, &save_path)?;
        } else if dyn_img.as_rgb8().is_some() {
            warp_black_extrap::<_, (u8, u8, u8), _, (u8, u8, u8), _>(dyn_img, motion, &save_path)?;
        } else if dyn_img.as_rgb16().is_some() {
            warp_black_extrap::<_, (u16, u16, u16), _, (u16, u16, u16), _>(
                dyn_img, motion, &save_path,
            )?;
        } else {
            panic!("Unknown image type");
        }

        pb.inc(1);
    }
    pb.finish();
    Ok(())
}

fn warp_black_extrap<Pix, T, V, O, I>(
    img: I,
    motion: Vector6<f32>,
    save_path: &Path,
) -> anyhow::Result<()>
where
    O: Scalar,
    DMatrix<O>: ToImage,
    V: Add<Output = V>,
    f32: Mul<V, Output = V>,
    T: Scalar + Copy + CanLinearInterpolate<V, O>,
    I: IntoDMatrix<Pix, T>,
{
    let mat = img.into_dmatrix();
    let warp_mat = registration::warp_black_extrap(&mat, &motion);

    // Save the image to disk.
    let warp_img = warp_mat.to_image();
    warp_img.save(save_path)?;
    Ok(())
}
