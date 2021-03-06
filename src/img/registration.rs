// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Registration algorithm for a sequence of slightly misaligned images.

use image::Primitive;
use nalgebra::{DMatrix, Matrix3, Matrix6, RealField, Scalar, Vector3, Vector6};
use std::ops::{Add, Mul};
use std::rc::Rc;

use crate::affine2d::{projection_mat, projection_params};
use crate::img::interpolation::CanLinearInterpolate;
use crate::interop::IntoImage;

/// Configuration (parameters) of the registration algorithm.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub lambda: f32,
    pub rho: f32,
    pub max_iterations: usize,
    pub threshold: f32,
    pub sparse_ratio_threshold: f32,
    pub levels: usize,
    pub trace: bool,
}

/// Type alias just to semantically differenciate Vec<Levels<_>> and Levels<Vec<_>>.
type Levels<T> = Vec<T>;

/// Trait for types that implement all the necessary stuff in order
/// to do registration on matrices of that type.
/// Basically u8 and u16 (only gray images supported for now).
pub trait CanRegister:
    Copy
    + Scalar
    + Primitive
    + crate::img::viz::ToRgb8
    + crate::img::multires::Bigger
    + crate::img::gradients::Bigger<<Self as CanRegister>::Bigger>
    + CanLinearInterpolate<f32, f32>
    + CanLinearInterpolate<f32, Self>
where
    DMatrix<Self>: IntoImage,
{
    type Bigger: Scalar + Copy + PartialOrd + Add<Output = Self::Bigger>;
}

impl CanRegister for u8 {
    type Bigger = u16;
}
impl CanRegister for u16 {
    type Bigger = u32;
}

/// Affine registration of single channel images.
///
/// Internally, this uses a multi-resolution approach,
/// where the motion vector computed at one resolution serves
/// as initialization for the next one.
///
/// The input images are passed by value to be used as the first level
/// of the multi-resolution pyramid.
pub fn gray_affine<T: CanRegister>(
    config: Config,
    imgs: Vec<DMatrix<T>>,
    sparse_diff_threshold: T::Bigger, // 50
) -> Result<(Vec<Vector6<f32>>, Vec<DMatrix<T>>), Box<dyn std::error::Error>>
where
    DMatrix<T>: IntoImage,
{
    // Get the number of images to align.
    let imgs_count = imgs.len();

    // Precompute a hierarchy of multi-resolution images and gradients norm.
    let mut multires_imgs: Vec<Levels<_>> = Vec::with_capacity(imgs_count);
    let mut multires_sparse_pixels: Vec<Levels<_>> = Vec::with_capacity(imgs_count);
    for im in imgs.into_iter() {
        let pyramid: Levels<DMatrix<T>> = crate::img::multires::mean_pyramid(config.levels, im);
        let gradients: Levels<DMatrix<T::Bigger>> = pyramid
            .iter()
            .map(crate::img::gradients::squared_norm_direct)
            .collect();
        let sparse_pixels = crate::img::sparse::select(sparse_diff_threshold, gradients.as_slice());
        multires_sparse_pixels.push(sparse_pixels);
        multires_imgs.push(pyramid);
    }

    // Save multires imgs.
    // crate::utils::save_imgs("out/multires_imgs", &multires_imgs[0]);

    // Save sparse pixels of first image.
    let mut multires_sparse_viz: Levels<DMatrix<(u8, u8, u8)>> = Vec::with_capacity(config.levels);
    for (sparse_mask, img_mat) in multires_sparse_pixels[0]
        .iter()
        .zip(multires_imgs[0].iter().rev())
    {
        multires_sparse_viz.push(crate::img::viz::mask_overlay(sparse_mask, img_mat));
    }
    // crate::utils::save_rgb_imgs::<&str, u8>(
    //     "out/multires_sparse_img0",
    //     multires_sparse_viz.as_slice(),
    // );

    // Transpose the `Vec<Levels<_>>` structure of multires images
    // into a `Levels<Vec<_>>` to have each level regrouped.
    let multires_imgs: Levels<Vec<_>> = crate::utils::transpose(multires_imgs);
    // let multires_sparse_pixels: Levels<Vec<_>> = crate::utils::transpose(multires_sparse_pixels);

    // // Merge sparse pixels by level.
    // let multires_sparse_pixels: Levels<_> = multires_sparse_pixels
    //     .iter()
    //     .map(|v| merge_sparse(v))
    //     .collect();
    let multires_sparse_pixels = multires_sparse_pixels[0].clone();

    // // Save merged sparse pixels of all images.
    // let mut multires_sparse_merged_viz = Vec::with_capacity(config.levels);
    // for (sparse_mask, img_mat) in multires_sparse_pixels
    //     .iter()
    //     .zip(multires_imgs.iter().rev().map(|v| &v[0]))
    // {
    //     multires_sparse_merged_viz.push(visualize_mask(sparse_mask, img_mat));
    // }
    // crate::utils::save_rgb_imgs("out/multires_sparse_merged", &multires_sparse_merged_viz);

    // Initialize the motion vector.
    let mut motion_vec = vec![Vector6::zeros(); imgs_count];

    // Multi-resolution algorithm.
    // Does the same thing at each level for the corresponding images and gradients.
    // The iterator is reversed to start at last level (lowest resolution).
    // Level 0 are the initial images.
    for (level, (lvl_imgs, lvl_sparse_pixels)) in multires_imgs
        .iter()
        .zip(multires_sparse_pixels.iter().rev())
        .enumerate()
        .rev()
    {
        eprintln!("\n=============  Start level {}  =============\n", level);

        // Algorithm parameters.
        let (height, width) = lvl_imgs[0].shape();
        let step_config = StepConfig {
            lambda: config.lambda,
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

        // Sparse filter.
        let pixels_count = height * width;
        let sparse_count: usize = lvl_sparse_pixels
            .iter()
            .map(|x| if *x { 1 } else { 0 })
            .sum();
        let sparse_ratio = sparse_count as f32 / pixels_count as f32;
        eprintln!(
            "Sparse ratio: {} / {} = {:.2}",
            sparse_count, pixels_count, sparse_ratio
        );

        // Choose sparsity.
        let sparsity: Sparsity;
        let actual_pixel_count: usize;
        let pixel_coordinates: Rc<Vec<(usize, usize)>>;
        if sparse_ratio > config.sparse_ratio_threshold {
            sparsity = Sparsity::Full;
            actual_pixel_count = pixels_count;
            pixel_coordinates = Rc::new(crate::utils::coords_col_major((height, width)).collect());
        } else {
            sparsity = Sparsity::Sparse;
            actual_pixel_count = sparse_count;
            pixel_coordinates = Rc::new(crate::utils::coordinates_from_mask(lvl_sparse_pixels));
        }

        // Declare mutable loop state.
        let mut loop_state;
        let mut imgs_registered;
        let obs = Obs {
            image_size: (width, height),
            images: lvl_imgs.as_slice(),
            sparsity,
            coordinates: pixel_coordinates.as_slice(),
        };

        // We also recompute the registered images before starting the algorithm loop.
        imgs_registered = DMatrix::zeros(actual_pixel_count, imgs_count);
        project_f32(
            pixel_coordinates.iter().cloned(),
            &mut imgs_registered,
            &lvl_imgs,
            &motion_vec,
        );

        // Updated state variables for the loops.
        loop_state = State {
            nb_iter: 0,
            imgs_registered,
            old_imgs_a: DMatrix::zeros(actual_pixel_count, imgs_count),
            errors: DMatrix::zeros(actual_pixel_count, imgs_count),
            lagrange_mult_rho: DMatrix::zeros(actual_pixel_count, imgs_count),
            motion_vec: motion_vec.clone(),
        };

        // Main loop.
        let mut continuation = Continue::Forward;
        while continuation == Continue::Forward {
            continuation = loop_state.step(&step_config, &obs);
        }

        // Update the motion vec before next level
        motion_vec = loop_state.motion_vec;
        // eprintln!("motion_vec:");
        // motion_vec.iter().for_each(|v| eprintln!("   {:?}", v.data));
    } // End of levels

    // Return the final motion vector.
    // And give back the images at original resolution.
    let imgs = multires_imgs.into_iter().next().unwrap();
    Ok((motion_vec, imgs))
}

/// Configuration parameters for the core loop of the algorithm.
struct StepConfig {
    lambda: f32,
    rho: f32,
    max_iterations: usize,
    threshold: f32,
    debug_trace: bool,
}

/// "Observations" contains the data provided outside the core of the algorithm.
/// These are immutable references since we are not supposed to mutate them.
struct Obs<'a, T: Scalar + Copy> {
    image_size: (usize, usize),
    images: &'a [DMatrix<T>],
    sparsity: Sparsity,
    coordinates: &'a [(usize, usize)],
}

enum Sparsity {
    Full,
    Sparse,
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
    imgs_registered: DMatrix<f32>,   // W(u; theta) in paper
    old_imgs_a: DMatrix<f32>,        // A in paper
    errors: DMatrix<f32>,            // e in paper
    lagrange_mult_rho: DMatrix<f32>, // y / rho in paper
    motion_vec: Vec<Vector6<f32>>,   // theta in paper
}

impl State {
    /// Core iteration step of the algorithm.
    fn step<T: Scalar + Copy + CanLinearInterpolate<f32, f32>>(
        &mut self,
        config: &StepConfig,
        obs: &Obs<T>,
    ) -> Continue {
        // Extract state variables to avoid prefixed notation later.
        let (width, height) = obs.image_size;
        let State {
            nb_iter,
            old_imgs_a,
            imgs_registered,
            errors,
            lagrange_mult_rho,
            motion_vec,
        } = self;
        // Pre-scale lambda.
        let lambda = config.lambda / (imgs_registered.nrows() as f32).sqrt();

        // A-update: low-rank approximation.
        let imgs_a_temp = &*imgs_registered + &*errors + &*lagrange_mult_rho;
        let mut svd = imgs_a_temp.svd(true, true);
        for x in svd.singular_values.iter_mut() {
            *x = shrink(1.0 / config.rho, *x);
        }
        let singular_values = svd.singular_values.clone();
        let imgs_a = svd.recompose().unwrap();

        // e-update: L1-regularized least-squares
        let errors_temp = &imgs_a - &*imgs_registered - &*lagrange_mult_rho;
        *errors = errors_temp.map(|x| shrink(lambda / config.rho, x));

        // theta-update: forwards compositional step of a Gauss-Newton approximation.
        let residuals = &errors_temp - &*errors;
        for i in 0..obs.images.len() {
            // Compute gradients for the registered image.
            let gradients = match &obs.sparsity {
                Sparsity::Full => compute_registered_gradients_full(
                    (height, width),
                    imgs_registered.column(i).as_slice(),
                ),
                Sparsity::Sparse => compute_registered_gradients_sparse(
                    &obs.images[i],
                    &(projection_mat(&motion_vec[i])),
                    obs.coordinates.iter().cloned(),
                )
                .collect(),
            };

            // Compute residuals and motion step.
            let step_params = forwards_compositional_step(
                (height, width),
                obs.coordinates.iter().cloned(),
                residuals.column(i).iter().cloned(),
                gradients.into_iter(),
            );

            // Save motion for this image.
            motion_vec[i] =
                projection_params(&(projection_mat(&motion_vec[i]) * projection_mat(&step_params)));
        }

        // Transform all motion parameters such that image 0 is the reference.
        let inverse_motion_ref = projection_mat(&motion_vec[0])
            .try_inverse()
            .expect("Error while inversing motion of reference image");
        for motion_params in motion_vec.iter_mut() {
            *motion_params =
                projection_params(&(inverse_motion_ref * projection_mat(&motion_params)));
        }

        // Update imgs_registered.
        project_f32(
            obs.coordinates.iter().cloned(),
            imgs_registered,
            &obs.images,
            &motion_vec,
        );

        // y-update: dual ascent
        *lagrange_mult_rho += &*imgs_registered - &imgs_a + &*errors;

        // Check convergence
        let residual = norm(&(&imgs_a - &*old_imgs_a)) / 1e-12.max(norm(old_imgs_a));
        if config.debug_trace {
            let nuclear_norm = singular_values.sum();
            let l1_norm = lambda * errors.map(|x| x.abs()).sum();
            let r = &*imgs_registered - &imgs_a + &*errors;
            let augmented_lagrangian = nuclear_norm
                + l1_norm
                + config.rho * (lagrange_mult_rho.component_mul(&r)).sum()
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
        if *nb_iter >= config.max_iterations || residual < config.threshold {
            continuation = Continue::Stop;
        }

        // Update state.
        *nb_iter += 1;
        *old_imgs_a = imgs_a;

        // Returned value.
        continuation
    }
}

fn compute_registered_gradients_full(shape: (usize, usize), registered: &[f32]) -> Vec<(f32, f32)> {
    let (nrows, ncols) = shape;
    let img_registered_shaped = DMatrix::from_iterator(nrows, ncols, registered.iter().cloned());
    crate::img::gradients::centered_f32(&img_registered_shaped)
        .data
        .into()
}

/// Compute the gradients of warped image.
/// There are more efficient ways than to interpolate 4 points,
/// but it would be to much trouble.
fn compute_registered_gradients_sparse<'a, T>(
    img: &'a DMatrix<T>,
    motion: &'a Matrix3<f32>,
    coordinates: impl Iterator<Item = (usize, usize)> + 'a,
) -> impl Iterator<Item = (f32, f32)> + 'a
where
    T: Scalar + Copy + CanLinearInterpolate<f32, f32>,
{
    coordinates.map(move |(x, y)| {
        // Horizontal gradient (gx).
        let x_left = x as f32 - 1.0;
        let x_right = x as f32 + 1.0;
        let new_left = motion * Vector3::new(x_left, y as f32, 1.0);
        let new_right = motion * Vector3::new(x_right, y as f32, 1.0);
        // WARNING: beware that interpolating with a f32 output normalize values in [0,1].
        let pixel_left: f32 = crate::img::interpolation::linear(new_left.x, new_left.y, img);
        let pixel_right: f32 = crate::img::interpolation::linear(new_right.x, new_right.y, img);

        // Vertical gradient (gy).
        let y_top = y as f32 - 1.0;
        let y_bot = y as f32 + 1.0;
        let new_top = motion * Vector3::new(x as f32, y_top, 1.0);
        let new_bot = motion * Vector3::new(x as f32, y_bot, 1.0);
        // WARNING: beware that interpolating with a f32 output normalize values in [0,1].
        let pixel_top: f32 = crate::img::interpolation::linear(new_top.x, new_top.y, img);
        let pixel_bot: f32 = crate::img::interpolation::linear(new_bot.x, new_bot.y, img);

        // Gradient.
        (
            0.5 * (pixel_right - pixel_left),
            0.5 * (pixel_bot - pixel_top),
        )
    })
}

fn forwards_compositional_step(
    shape: (usize, usize),
    coordinates: impl Iterator<Item = (usize, usize)>,
    residuals: impl Iterator<Item = f32>,
    gradients: impl Iterator<Item = (f32, f32)>,
) -> Vector6<f32> {
    let (height, width) = shape;
    let mut descent_params = Vector6::zeros();
    let mut hessian = Matrix6::zeros();
    let border = (0.04 * height.min(width) as f32) as usize;
    for (((x, y), res), (gx, gy)) in coordinates.zip(residuals).zip(gradients) {
        // Only use points within a given margin.
        if x > border && x + border < width && y > border && y + border < height {
            let x_ = x as f32;
            let y_ = y as f32;
            let jac_t = Vector6::new(x_ * gx, x_ * gy, y_ * gx, y_ * gy, gx, gy);
            hessian += jac_t * jac_t.transpose();
            descent_params += res * jac_t;
        }
    }
    let hessian_chol = hessian.cholesky().expect("Error hessian choleski");
    hessian_chol.solve(&descent_params)
}

/// Compute the projection of each pixel of the image (modify in place).
/// CAREFUL: coordinates must have the same amount of items that
/// the number of rows in registered.
/// Otherwise it may silently compute a wrong projection.
/// I don't know how to assert the number of items in the coordinates iterator.
fn project_f32<T: Scalar + Copy + CanLinearInterpolate<f32, f32>>(
    coordinates: impl Iterator<Item = (usize, usize)> + Clone,
    registered: &mut DMatrix<f32>,
    imgs: &[DMatrix<T>],
    motion_vec: &[Vector6<f32>],
) {
    for (i, motion) in motion_vec.iter().enumerate() {
        let motion_mat = projection_mat(motion);
        let mut registered_col = registered.column_mut(i);
        for ((x, y), pixel) in coordinates.clone().zip(registered_col.iter_mut()) {
            let new_pos = motion_mat * Vector3::new(x as f32, y as f32, 1.0);
            // WARNING: beware that interpolating with a f32 output normalize values in [0,1].
            let interp: f32 = crate::img::interpolation::linear(new_pos.x, new_pos.y, &imgs[i]);
            *pixel = interp;
        }
    }
}

/// Compute the projection of each pixel of the image.
pub fn reproject<T, V, O>(imgs: &[DMatrix<T>], motion_vec: &[Vector6<f32>]) -> Vec<DMatrix<O>>
where
    O: Scalar,
    V: Add<Output = V>,
    f32: Mul<V, Output = V>,
    T: Scalar + Copy + CanLinearInterpolate<V, O>,
{
    let warp_pair = |(im, motion)| warp(im, motion);
    imgs.iter().zip(motion_vec).map(warp_pair).collect()
}

pub fn warp<T, V, O>(img: &DMatrix<T>, motion_params: &Vector6<f32>) -> DMatrix<O>
where
    O: Scalar,
    V: Add<Output = V>,
    f32: Mul<V, Output = V>,
    T: Scalar + Copy + CanLinearInterpolate<V, O>,
{
    let (nrows, ncols) = img.shape();
    let motion_mat = projection_mat(motion_params);
    DMatrix::from_fn(nrows, ncols, |i, j| {
        let new_pos = motion_mat * Vector3::new(j as f32, i as f32, 1.0);
        crate::img::interpolation::linear(new_pos.x, new_pos.y, img)
    })
}

/// Computes the sqrt of the sum of squared values.
/// This is the L2 norm of the vectorized version of the matrix.
fn norm(matrix: &DMatrix<f32>) -> f32 {
    norm_sqr(matrix).sqrt() as f32
}

/// Computes the sum of squared values of a matrix.
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
