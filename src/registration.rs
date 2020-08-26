use nalgebra::{DMatrix, DVector, Matrix2, RealField, Vector2};

#[derive(Debug)]
pub struct Config {
    pub do_registration: bool,
    pub do_image_correction: bool,
    pub lambda: f32,
    pub rho: f32,
    pub max_iterations: usize,
    pub threshold: f32,
    pub image_max: f32,
    pub trace: bool,
}

struct StepConfig {
    do_image_correction: bool,
    lambda: f32,
    rho: f32,
    max_iterations: usize,
    threshold: f32,
    debug_trace: bool,
}

struct Obs<'a> {
    image_size: (usize, usize),
    images: &'a [DMatrix<f32>],
    gradients: &'a [(DMatrix<f32>, DMatrix<f32>)],
    hessians_inv: &'a [Matrix2<f32>],
}

#[derive(PartialEq)]
enum Continue {
    Forward,
    Stop,
}

struct State {
    imgs_registered: DMatrix<f32>,
    old_imgs_hat: DMatrix<f32>,
    errors: DMatrix<f32>,
    lagrange_mult: DMatrix<f32>,
    motion_vec: Vec<Vector2<f32>>,
}

type Levels<T> = Vec<T>;

pub fn gray_images(
    config: Config,
    imgs: Vec<DMatrix<u8>>,
) -> Result<Vec<Vector2<f32>>, Box<dyn std::error::Error>> {
    let nb_imgs = imgs.len();

    // Precompute multi-resolution images.
    let multires_imgs: Vec<Levels<_>> = imgs
        .into_iter()
        .map(|im| crate::multires::mean_pyramid(2, im))
        .collect();

    // Precompute multi-resolution gradients.
    let multires_gradients: Vec<Levels<_>> = multires_imgs
        .iter()
        .map(|multi| {
            multi
                .iter()
                .map(|im| crate::gradients::centered(&im))
                .collect()
        })
        .collect();

    // Transpose multires images and gradients to have each level in its own vec.
    let multires_imgs: Levels<Vec<_>> = crate::utils::transpose(multires_imgs);
    let multires_gradients: Levels<Vec<_>> = crate::utils::transpose(multires_gradients);

    // Initialize the motion vector.
    let mut motion_vec = vec![Vector2::zeros(); nb_imgs];

    // Multi-resolution algorithm.
    for (level, (l_imgs, l_gradients)) in multires_imgs
        .iter()
        .zip(multires_gradients.iter())
        .enumerate()
        // Reverse iterator to start at lowest resolution.
        .rev()
    {
        eprintln!("\n=============  Start level {}  =============\n", level);

        // Build f32 float versions of the images and gradients.
        let u8_to_float = |m: &DMatrix<u8>| m.map(|x| x as f32 / config.image_max);
        let imgs_f32: Vec<DMatrix<f32>> = l_imgs.iter().map(u8_to_float).collect();

        let i16_to_float = |m: &DMatrix<i16>| m.map(|x| x as f32 / config.image_max);
        let gradients_f32: Vec<(DMatrix<f32>, DMatrix<f32>)> = l_gradients
            .iter()
            .map(|(gx, gy)| (i16_to_float(gx), i16_to_float(gy)))
            .collect();

        // Precompute inverse of hessian matrices.
        let hessians_inv: Vec<_> = gradients_f32
            .iter()
            .map(|(gradients_x, gradients_y)| {
                let mut hessian = Matrix2::zeros();
                for (gx, gy) in gradients_x.iter().zip(gradients_y.iter()) {
                    hessian += Matrix2::new(gx * gx, gx * gy, gx * gy, gy * gy);
                }
                hessian
                    .try_inverse()
                    .expect("Error while inverting hessian")
            })
            .collect();

        // Algorithm parameters.
        let (height, width) = imgs_f32[0].shape();
        let obs = Obs {
            image_size: (width, height),
            images: &imgs_f32,
            gradients: &gradients_f32,
            hessians_inv: &hessians_inv,
        };
        let step_config = StepConfig {
            do_image_correction: config.do_image_correction,
            // Scale lambda by the number of pixels.
            lambda: config.lambda / ((width * height) as f32).sqrt(),
            rho: config.rho,
            max_iterations: config.max_iterations,
            threshold: config.threshold,
            debug_trace: config.trace,
        };

        // Initialize loop variables.
        // motion_vec is doubled when changing level.
        for motion in motion_vec.iter_mut() {
            *motion *= 2.0;
        }
        let mut imgs_registered = DMatrix::zeros(height * width, nb_imgs);
        reproject_f32(width, height, &mut imgs_registered, &imgs_f32, &motion_vec);
        let mut state = State {
            imgs_registered,
            old_imgs_hat: DMatrix::zeros(height * width, nb_imgs),
            errors: DMatrix::zeros(height * width, nb_imgs),
            lagrange_mult: DMatrix::zeros(height * width, nb_imgs),
            motion_vec: motion_vec.clone(),
        };

        // Main loop.
        let mut nb_iter = 0;
        let mut continuation = Continue::Forward;
        while continuation == Continue::Forward {
            let (new_state, new_continuation) = step(&step_config, &obs, nb_iter, state);
            nb_iter += 1;
            state = new_state;
            continuation = new_continuation;
        }

        // Update the motion vec before next level
        motion_vec = state.motion_vec;
        eprintln!("motion_vec:");
        motion_vec.iter().for_each(|v| eprintln!("   {:?}", v.data));
    } // End of levels

    // Return the final motion vector.
    Ok(motion_vec)
}

/// Core iteration step of the algorithm.
fn step(config: &StepConfig, obs: &Obs, nb_iter: usize, state: State) -> (State, Continue) {
    // Retrieve state variables.
    let (width, height) = obs.image_size;
    let State {
        old_imgs_hat,
        mut imgs_registered,
        mut errors,
        mut lagrange_mult,
        mut motion_vec,
    } = state;

    // A-update: low-rank approximation
    let imgs_hat_temp = &imgs_registered + &errors + &lagrange_mult / config.rho;
    let mut svd = imgs_hat_temp.svd(true, true);
    for x in svd.singular_values.iter_mut() {
        *x = shrink(1.0 / config.rho, *x);
    }
    let singular_values = svd.singular_values.clone();
    let imgs_hat = svd.recompose().unwrap();

    // e-update: L1-regularized least-squares
    if config.do_image_correction {
        errors = &imgs_hat - &imgs_registered - &lagrange_mult / config.rho;
        for x in errors.iter_mut() {
            *x = shrink(config.lambda / config.rho, *x);
        }
    }

    // v-update: inverse compositional step.
    for (i, (ux, uy)) in obs.gradients.iter().enumerate() {
        // Use the first image as the reference frame.
        if i == 0 {
            continue;
        }
        let mut new_image = crate::utils::reshape(
            imgs_hat.column(i) - lagrange_mult.column(i) / config.rho - errors.column(i),
            height,
            width,
        );
        let dx = motion_vec[i].x;
        let dy = motion_vec[i].y;
        new_image = DMatrix::from_fn(height, width, |i, j| {
            crate::interpolation::linear(j as f32 - dx, i as f32 - dy, &new_image)
        });
        let residuals = &new_image - &obs.images[i];
        let mut g = Vector2::zeros();
        ux.iter()
            .zip(uy.iter())
            .zip(residuals.iter())
            .for_each(|((&gx, &gy), &res)| {
                g += res * Vector2::new(gx, gy);
            });
        let motion_step = &obs.hessians_inv[i] * g;

        // Save motion for this image.
        // We should do -= since we prepend the inverse
        // but since we use motion_vec on U later for registration,
        // we might as well re-inverse it on the fly here.
        motion_vec[i] += motion_step;
    }

    reproject_f32(
        width,
        height,
        &mut imgs_registered,
        &obs.images,
        &motion_vec,
    );

    // w-update: dual ascent
    lagrange_mult += config.rho * (&imgs_registered - &imgs_hat + &errors);

    // Check convergence
    let residual = norm(&(&imgs_hat - &old_imgs_hat)) / 1e-12.max(norm(&old_imgs_hat));
    if config.debug_trace {
        let nuclear_norm = singular_values.sum();
        let l1_norm = config.lambda * errors.map(|x| x.abs()).sum();
        let r = &imgs_registered - &imgs_hat + &errors;
        let augmented_lagrangian = nuclear_norm
            + l1_norm
            + (lagrange_mult.component_mul(&r)).sum()
            + 0.5 * config.rho * (norm_sqr(&r) as f32);
        eprintln!("");
        eprintln!("Iteration {}:", nb_iter);
        eprintln!("    Nucl norm: {}", nuclear_norm);
        eprintln!("    L1 norm: {}", l1_norm);
        eprintln!("    Nucl + L1: {}", l1_norm + nuclear_norm);
        eprintln!("    Aug. Lagrangian: {}", augmented_lagrangian);
        eprintln!("    residual: {}", residual);
        eprintln!("");
    }
    let mut continuation = Continue::Forward;
    if nb_iter >= config.max_iterations || residual < config.threshold {
        continuation = Continue::Stop;
    }
    (
        State {
            imgs_registered,
            old_imgs_hat: imgs_hat,
            errors,
            lagrange_mult,
            motion_vec,
        },
        continuation,
    )
}

/// Recompute the registered image.
fn reproject_f32(
    width: usize,
    height: usize,
    registered: &mut DMatrix<f32>,
    imgs: &[DMatrix<f32>],
    motion_vec: &[Vector2<f32>],
) {
    for (i, motion) in motion_vec.iter().enumerate() {
        let dx = motion.x;
        let dy = motion.y;
        let mut idx = 0;
        for x in 0..width {
            for y in 0..height {
                registered[(idx, i)] =
                    crate::interpolation::linear(x as f32 + dx, y as f32 + dy, &imgs[i]);
                idx += 1;
            }
        }
    }
}

/// Generate final registered images.
pub fn reproject_u8(imgs: &[DMatrix<u8>], motion_vec: &[Vector2<f32>]) -> Vec<DMatrix<u8>> {
    let (height, width) = imgs[0].shape();
    let mut all_registered = Vec::new();
    for (im, motion) in imgs.iter().zip(motion_vec.iter()) {
        let dx = motion.x;
        let dy = motion.y;
        let registered = DMatrix::from_fn(height, width, |i, j| {
            crate::interpolation::linear(j as f32 + dx, i as f32 + dy, im)
                .max(0.0)
                .min(255.0) as u8
        });
        all_registered.push(registered);
    }
    all_registered
}

/// Computes the sqrt of the sum of squared values.
/// This is the L2 norm of the vectorized version of the matrix.
fn norm(matrix: &DMatrix<f32>) -> f32 {
    norm_sqr(matrix).sqrt() as f32
}

fn norm_sqr(matrix: &DMatrix<f32>) -> f64 {
    matrix.iter().map(|&x| (x as f64).powi(2)).sum()
}

fn shrink<T: RealField>(alpha: T, x: T) -> T {
    let alpha = alpha.abs();
    if x.is_sign_positive() {
        (x - alpha).max(T::zero())
    } else {
        (x + alpha).min(T::zero())
    }
}
