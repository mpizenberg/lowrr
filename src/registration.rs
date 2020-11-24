// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Registration algorithm for a sequence of slightly misaligned images.

use nalgebra::{DMatrix, Matrix3, Matrix6, RealField, Vector2, Vector3, Vector6};

/// Configuration (parameters) of the registration algorithm.
#[derive(Debug)]
pub struct Config {
    pub do_image_correction: bool,
    pub lambda: f32,
    pub rho: f32,
    pub max_iterations: usize,
    pub threshold: f32,
    pub image_max: f32,
    pub levels: usize,
    pub trace: bool,
}

/// Type alias just to visually differenciate Vec<Vec<_>>
/// when it is Vec<Levels<_>> or Levels<Vec<_>>.
type Levels<T> = Vec<T>;

/// Registration of single channel images.
///
/// Internally, this uses a multi-resolution approach,
/// where the motion vector computed at one resolution serves
/// as initialization for the next one.
pub fn gray_images(
    config: Config,
    imgs: Vec<DMatrix<u8>>,
) -> Result<Vec<Vector6<f32>>, Box<dyn std::error::Error>> {
    // Get the number of images to align.
    let nb_imgs = imgs.len();

    // Precompute a hierarchy of multi-resolution images.
    let multires_imgs: Vec<Levels<_>> = imgs
        .into_iter()
        .map(|im| crate::multires::mean_pyramid(config.levels, im))
        .collect();

    // Precompute multi-resolution gradients.
    // let kernel = crate::filter::gaussian_kernel(1.0, 5);
    let multires_gradients: Vec<Levels<_>> = multires_imgs
        .iter()
        .map(|multi| {
            multi
                .iter()
                // .map(|im| crate::filter::conv_2d_direct_same(im, &kernel))
                .map(|im| crate::gradients::centered(&im))
                .collect()
        })
        .collect();

    // Transpose the `Vec<Levels<_>>` structure of multires images and gradients
    // into a `Levels<Vec<_>>` to have each level in its own vec.
    let multires_imgs: Levels<Vec<_>> = crate::utils::transpose(multires_imgs);
    let multires_gradients: Levels<Vec<_>> = crate::utils::transpose(multires_gradients);

    // Initialize the motion vector.
    let mut motion_vec = vec![Vector6::zeros(); nb_imgs];

    // Multi-resolution algorithm.
    // Does the same thing at each level for the corresponding images and gradients.
    // The iterator is reversed to start at last level (lowest resolution).
    // Level 0 are the initial images.
    for (level, (l_imgs, l_gradients)) in multires_imgs
        .iter()
        .zip(multires_gradients.iter())
        .enumerate()
        .rev()
    {
        eprintln!("\n=============  Start level {}  =============\n", level);

        // Build f32 float versions of the images and gradients.
        // Normalize values to [0..1].
        let u8_to_float = |m: &DMatrix<u8>| m.map(|x| x as f32 / config.image_max);
        let imgs_f32: Vec<DMatrix<f32>> = l_imgs.iter().map(u8_to_float).collect();

        let i16_to_float = |m: &DMatrix<i16>| m.map(|x| x as f32 / config.image_max);
        let gradients_f32: Vec<(DMatrix<f32>, DMatrix<f32>)> = l_gradients
            .iter()
            .map(|(gx, gy)| (i16_to_float(gx), i16_to_float(gy)))
            .collect();

        // Precompute the Jacobian for an affine model
        // | 1+a  c   e |
        // |  b  1+d  f |
        //
        // J = | x*gx  x*gy  y*gx  y*gy  gx  gy |
        let jacobians_transposed_all: Vec<Vec<_>> = gradients_f32
            .iter()
            .map(|(gradients_x, gradients_y)| {
                // In one image:
                let (height, _) = gradients_x.shape();
                let mut x = 0;
                let mut y = 0;
                gradients_x
                    .iter()
                    .zip(gradients_y.iter())
                    .map(|(&gx, &gy)| {
                        let jac = Vector6::new(
                            x as f32 * gx,
                            x as f32 * gy,
                            y as f32 * gx,
                            y as f32 * gy,
                            gx,
                            gy,
                        );
                        y += 1;
                        if y == height {
                            x += 1;
                            y = 0;
                        }
                        jac
                    })
                    .collect()
            })
            .collect();

        // Precompute inverse of Hessian matrices.
        // They will be used later to estimate motion steps.
        //
        // H = J^t * J
        //
        // Where J is the NxM Jacobian
        // N: number of model parameters
        // M: number of pixels in an image
        let hessians_inv: Vec<_> = jacobians_transposed_all
            .iter()
            .map(|jac_t_vec| {
                let mut x = 0;
                let mut y = 0;
                let mut hessian = Matrix6::zeros();
                let (height, width) = l_imgs[0].shape();
                let border = (0.02 * height as f32) as usize;
                for jac_t in jac_t_vec.iter() {
                    if x > border && x + border < width {
                        if y > border && y + border < height {
                            hessian += jac_t * jac_t.transpose();
                        }
                    }
                    y += 1;
                    if y >= height {
                        y = 0;
                        x += 1;
                    }
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
            jacobians_transposed_all: &jacobians_transposed_all,
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

        // motion_vec is adapted when changing level.
        for motion in motion_vec.iter_mut() {
            motion[4] = 2.0 * motion[4];
            motion[5] = 2.0 * motion[5];
        }

        // We also recompute the registered images before starting the algorithm loop.
        let mut imgs_registered = DMatrix::zeros(height * width, nb_imgs);
        reproject_f32(width, height, &mut imgs_registered, &imgs_f32, &motion_vec);

        // Updated state variables for the loops.
        let mut loop_state = State {
            nb_iter: 0,
            imgs_registered,
            old_imgs_hat: DMatrix::zeros(height * width, nb_imgs),
            errors: DMatrix::zeros(height * width, nb_imgs),
            lagrange_mult: DMatrix::zeros(height * width, nb_imgs),
            motion_vec: motion_vec.clone(),
        };

        // Main loop.
        let mut continuation = Continue::Forward;
        while continuation == Continue::Forward {
            let (new_state, new_continuation) = step(&step_config, &obs, loop_state);
            loop_state = new_state;
            continuation = new_continuation;
        }

        // Update the motion vec before next level
        motion_vec = loop_state.motion_vec;
        eprintln!("motion_vec:");
        motion_vec.iter().for_each(|v| eprintln!("   {:?}", v.data));
    } // End of levels

    // Return the final motion vector.
    Ok(motion_vec)
}

/// Configuration parameters for the core loop of the algorithm.
struct StepConfig {
    do_image_correction: bool,
    lambda: f32,
    rho: f32,
    max_iterations: usize,
    threshold: f32,
    debug_trace: bool,
}

/// "Observations" contains the data provided outside the core of the algorithm.
/// These are immutable references since we are not supposed to mutate them.
struct Obs<'a> {
    image_size: (usize, usize),
    images: &'a [DMatrix<f32>],
    gradients: &'a [(DMatrix<f32>, DMatrix<f32>)],
    jacobians_transposed_all: &'a [Vec<Vector6<f32>>],
    hessians_inv: &'a [Matrix6<f32>],
}

/// Simple enum type to indicate if we should continue to loop.
/// This is to avoid the ambiguity of booleans.
#[derive(PartialEq)]
enum Continue {
    Forward,
    Stop,
}

/// State variables of the loop.
struct State {
    nb_iter: usize,
    imgs_registered: DMatrix<f32>, // W(u; theta) in paper
    old_imgs_hat: DMatrix<f32>,    // A in paper
    errors: DMatrix<f32>,          // e in paper
    lagrange_mult: DMatrix<f32>,   // w in paper
    motion_vec: Vec<Vector6<f32>>, // theta in paper
}

/// Core iteration step of the algorithm.
fn step(config: &StepConfig, obs: &Obs, state: State) -> (State, Continue) {
    // Extract state variables to avoid prefixed notation later.
    let (width, height) = obs.image_size;
    let State {
        nb_iter,
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

    // v-update: inverse compositional step of a Gauss-Newton approximation.
    for i in 0..obs.images.len() {
        // Use the first image as the reference frame.
        // So we skip the iteration 0 such that motions_vec[0] is [0, 0].
        if i == 0 {
            // continue;
        }

        // Compute the image that we want to be aligned with.
        let mut new_image = crate::utils::reshape(
            imgs_hat.column(i) - lagrange_mult.column(i) / config.rho - errors.column(i),
            height,
            width,
        );

        // Reproject that image back to something near to our original image.
        // This is important to upheld the "small displacement" requirement
        // for our Gauss-Newton approximation of Jacobians.
        //
        // Beware that we use (-dx, -dy) since motion_vec stores the motion
        // required to align the original images, so this is the opposite.
        let motion_mat = projection_mat(&motion_vec[i])
            .try_inverse()
            .expect("woops inverse projection matrix");
        new_image = DMatrix::from_fn(height, width, |i, j| {
            let new_pos = motion_mat * Vector3::new(j as f32, i as f32, 1.0);
            crate::interpolation::linear(new_pos.x, new_pos.y, &new_image)
        });

        // Compute residuals and motion step,
        // following the inverse compositional scheme (CF Baker and Matthews).
        let residuals = &new_image - &obs.images[i];
        let jac_t_vec = &obs.jacobians_transposed_all[i];
        let mut g = Vector6::zeros();
        let mut x = 0;
        let mut y = 0;
        let border = (0.02 * height as f32) as usize;
        jac_t_vec
            .iter()
            .zip(residuals.iter())
            .for_each(|(&jac_t, &res)| {
                if x > border && x + border < width {
                    if y > border && y + border < height {
                        g += res * jac_t;
                    }
                }
                y += 1;
                if y >= height {
                    y = 0;
                    x += 1;
                }
            });
        let step_params = &obs.hessians_inv[i] * g;

        // Save motion for this image.
        // We should prepend the inverse of the motion step,
        // but since we use motion_vec on U later for registration,
        // we might as well re-inverse it on the fly here.
        motion_vec[i] =
            projection_params(&(projection_mat(&step_params) * projection_mat(&motion_vec[i])));
    }

    // Transform all motion parameters such that image 0 is the reference.
    let inverse_motion_ref = projection_mat(&motion_vec[0])
        .try_inverse()
        .expect("Error while inversing motion of reference image");
    for motion_params in motion_vec.iter_mut() {
        *motion_params = projection_params(&(inverse_motion_ref * projection_mat(&motion_params)));
    }

    // Update imgs_registered.
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

    // Returned value
    (
        State {
            nb_iter: nb_iter + 1,
            imgs_registered,
            old_imgs_hat: imgs_hat,
            errors,
            lagrange_mult,
            motion_vec,
        },
        continuation,
    )
}

#[rustfmt::skip]
pub fn projection_mat(params: &Vector6<f32>) -> Matrix3<f32> {
    Matrix3::new(
        1.0 + params[0], params[2], params[4],
        params[1], 1.0 + params[3], params[5],
        0.0, 0.0, 1.0,
    )
}

pub fn projection_params(mat: &Matrix3<f32>) -> Vector6<f32> {
    Vector6::new(
        mat.m11 - 1.0,
        mat.m21,
        mat.m12,
        mat.m22 - 1.0,
        mat.m13,
        mat.m23,
    )
}

/// Recompute the registered image (modify in place).
fn reproject_f32(
    width: usize,
    height: usize,
    registered: &mut DMatrix<f32>,
    imgs: &[DMatrix<f32>],
    motion_vec: &[Vector6<f32>],
) {
    for (i, motion) in motion_vec.iter().enumerate() {
        let motion_mat = projection_mat(motion);
        let mut idx = 0;
        for x in 0..width {
            for y in 0..height {
                let new_pos = motion_mat * Vector3::new(x as f32, y as f32, 1.0);
                registered[(idx, i)] = crate::interpolation::linear(new_pos.x, new_pos.y, &imgs[i]);
                idx += 1;
            }
        }
    }
}

/// Generate final registered images.
pub fn reproject_u8(imgs: &[DMatrix<u8>], motion_vec: &[Vector6<f32>]) -> Vec<DMatrix<u8>> {
    let (height, width) = imgs[0].shape();
    let mut all_registered = Vec::new();
    for (im, motion) in imgs.iter().zip(motion_vec.iter()) {
        let motion_mat = projection_mat(motion);
        let registered = DMatrix::from_fn(height, width, |i, j| {
            let new_pos = motion_mat * Vector3::new(j as f32, i as f32, 1.0);
            crate::interpolation::linear(new_pos.x, new_pos.y, im)
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

/// Shrink values toward 0.
fn shrink<T: RealField>(alpha: T, x: T) -> T {
    let alpha = alpha.abs();
    if x.is_sign_positive() {
        (x - alpha).max(T::zero())
    } else {
        (x + alpha).min(T::zero())
    }
}
