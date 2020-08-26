use lowrr::interop;
use lowrr::registration;
mod unused;

use glob::glob;
use nalgebra::DMatrix;
use std::path::{Path, PathBuf};

// Default values for some of the program arguments.
const DEFAULT_OUT_DIR: &str = "out";
const DEFAULT_LEVELS: usize = 4;
const DEFAULT_LAMBDA: f32 = 1.5;
const DEFAULT_RHO: f32 = 0.1;
const DEFAULT_THRESHOLD: f32 = 1e-3;
const DEFAULT_MAX_ITERATIONS: usize = 20;
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

USAGE:
    lowrr [FLAGS] IMAGE_FILES
    For example:
        lowrr --trace *.png

FLAGS:
    --help                 # Print this message and exit
    --version              # Print version and exit
    --out-dir dir/         # Output directory to write results (default: {})
    --trace                # Print some debug output while running
    --levels int           # Number of levels for the multi-resolution approach (default: {})
    --no-image-correction  # Avoid image correction
    --lambda float         # Weight of the L1 term (high means no correction) (default: {})
    --rho float            # Lagrangian penalty (default: {})
    --threshold float      # Stop when relative diff between two estimate of corrected image falls below this (default: {})
    --max-iterations int   # Maximum number of iterations (default: {})
    --image-max float      # Maximum possible value of the images for scaling (default: {})
"#,
        DEFAULT_OUT_DIR,
        DEFAULT_LEVELS,
        DEFAULT_LAMBDA,
        DEFAULT_RHO,
        DEFAULT_THRESHOLD,
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
}

/// Function parsing the command line arguments and returning an Args object or an error.
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = pico_args::Arguments::from_env();

    // Retrieve command line arguments.
    let help = args.contains(["-h", "--help"]);
    let version = args.contains(["-v", "--version"]);
    let do_image_correction = !args.contains("--no-image-correction");
    let trace = args.contains("--trace");
    let lambda = args
        .opt_value_from_str("--lambda")?
        .unwrap_or(DEFAULT_LAMBDA);
    let rho = args.opt_value_from_str("--rho")?.unwrap_or(DEFAULT_RHO);
    let threshold = args
        .opt_value_from_str("--threshold")?
        .unwrap_or(DEFAULT_THRESHOLD);
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
            max_iterations,
            levels,
            image_max,
        },
        help,
        version,
        out_dir,
        images_paths,
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
    let (dataset, _) = load_dataset(&args.images_paths)?;

    // Use the algorithm corresponding to the type of data.
    match dataset {
        Dataset::GrayImages(imgs) => {
            // Compute the motion of each image for registration.
            let motion_vec = registration::gray_images(args.config, imgs.clone())?;

            // Reproject (interpolation + extrapolation) images according to that motion.
            let registered_imgs = registration::reproject_u8(&imgs, &motion_vec);

            // Write the registered images to the output directory.
            std::fs::create_dir_all(&out_dir_path)
                .expect(&format!("Could not create output dir: {:?}", &out_dir_path));
            registered_imgs.iter().enumerate().for_each(|(i, img)| {
                interop::image_from_matrix(img)
                    .save(&out_dir_path.join(format!("{}.png", i)))
                    .expect("Error saving image");
            });
            Ok(())
        }
        Dataset::RgbImages { red, green, blue } => unimplemented!(),
        Dataset::RawImages(imgs) => unimplemented!(),
    }
}

enum Dataset {
    RawImages(Vec<DMatrix<u16>>),
    GrayImages(Vec<DMatrix<u8>>),
    RgbImages {
        red: Vec<DMatrix<u8>>,
        green: Vec<DMatrix<u8>>,
        blue: Vec<DMatrix<u8>>,
    },
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
        let images: Vec<DMatrix<u8>> = paths
            .iter()
            .map(|path| image::open(path).unwrap())
            // Temporary convert color to gray.
            // .map(|i| i.into_luma())
            // .map(interop::matrix_from_image)
            .map(|i| i.into_rgb())
            .map(interop::matrix_from_rgb_image)
            // Temporary only keep red channel.
            .map(|m| m.map(|(_red, green, _blue)| green))
            .collect();
        let (height, width) = images[0].shape();
        Ok((Dataset::GrayImages(images), (width, height)))
    } else {
        panic!("There is a mix of image types")
    }
}
