use nalgebra::base::dimension::{Dynamic, U2};
use nalgebra::base::Scalar;
use nalgebra::{DMatrix, DVector, MatrixMN, RealField, Vector2};

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

pub fn gray_images(
    config: Config,
    width: usize,
    height: usize,
    imgs: &[DMatrix<u8>],
) -> Result<(Vec<DMatrix<u8>>, Vec<Vector2<f32>>), Box<dyn std::error::Error>> {
    // Precompute image gradients on smoothed images.
    let imgs_gradients: Vec<_> = imgs
        .iter()
        // .map(smooth)
        .map(|im| crate::gradients::centered(&im))
        .collect();

    // Debugging trace.
    if config.trace {
        let u_f32 = mat_from_vec(height, width, &|&x| x as f32, &imgs);
        let svd0 = u_f32.svd(false, false);
        println!("===================================");
        println!("Initial nucl_norm: {:?}", svd0.singular_values.sum());
    }

    // Scale lambda by the number of pixels.
    let lambda = config.lambda / ((width * height) as f32).sqrt();

    // Scaling factor to bring image values in [0..1].
    let imax_inv = 1.0 / config.image_max;

    // Initialize loop variables.
    let nb_imgs = imgs.len();
    let mut imgs_registered = mat_from_vec(height, width, &|&x| x as f32 * imax_inv, &imgs);
    let mut old_imgs_hat = DMatrix::<f32>::zeros(height * width, nb_imgs);
    let mut errors = DMatrix::<f32>::zeros(height * width, nb_imgs);
    let mut lagrange_mult = DMatrix::<f32>::zeros(height * width, nb_imgs);
    let mut motion_vec = vec![Vector2::zeros(); nb_imgs];

    // Main loop.
    for iteration in 0..config.max_iterations {
        if config.trace {
            println!("===================================");
        }

        // A-update: low-rank approximation
        let imgs_step = &imgs_registered + &errors + &lagrange_mult / config.rho;
        let mut svd = imgs_step.svd(true, true);
        for x in svd.singular_values.iter_mut() {
            *x = shrink(1.0 / config.rho, *x);
        }
        let imgs_hat = svd.clone().recompose()?;

        // e-update: L1-regularized least-squares
        if config.do_image_correction {
            errors = &imgs_hat - &imgs_registered - &lagrange_mult / config.rho;
            for x in errors.iter_mut() {
                *x = shrink(lambda / config.rho, *x);
            }
        }

        // v-update: linear least-squares registration
        if config.do_registration {
            for (i, (ux, uy)) in imgs_gradients.iter().enumerate() {
                #[allow(non_snake_case)]
                let Ai = MatrixMN::<f32, Dynamic, U2>::from_iterator(
                    height * width,
                    ux.iter().chain(uy.iter()).map(|&x| x as f32),
                );
                let img = DVector::from_iterator(height * width, imgs[i].iter().map(|&x| x as f32));
                let bi = imgs_hat.column(i)
                    - lagrange_mult.column(i) / config.rho
                    - img // There is a shape issue here right?
                    - errors.column(i);
                #[allow(non_snake_case)]
                let Ai_svd = Ai.svd(true, true);
                let motion = Ai_svd.solve(&bi, 1e-12)?;
                let dx = motion.x;
                let dy = motion.y;
                motion_vec[i] = motion;
                let mut idx = 0;
                for x in 0..width {
                    for y in 0..height {
                        imgs_registered[(idx, i)] = imax_inv
                            * crate::interpolation::linear(x as f32 + dx, y as f32 + dy, &imgs[i]);
                        idx += 1;
                    }
                }
            }
        }

        // w-update: dual ascent
        lagrange_mult += config.rho * (&imgs_registered - &imgs_hat + &errors);

        // Check convergence
        let residual = norm(&(&imgs_hat - &old_imgs_hat)) / 1e-12.max(norm(&old_imgs_hat));
        if config.trace {
            let nuclear_norm = svd.singular_values.sum();
            let l1_norm = lambda * errors.map(|x| x.abs()).sum();
            let r = &imgs_registered - &imgs_hat + &errors;
            let augmented_lagrangian = nuclear_norm
                + l1_norm
                + (lagrange_mult.component_mul(&r)).sum()
                + 0.5 * config.rho * (norm_sqr(&r) as f32);
            println!("Iteration {} - Nucl norm: {}", iteration, nuclear_norm);
            println!("Iteration {} - L1 norm: {}", iteration, l1_norm);
            println!(
                "Iteration {} - Nucl + L1: {}",
                iteration,
                l1_norm + nuclear_norm
            );
            println!(
                "Iteration {} - Aug. Lagrangian: {}",
                iteration, augmented_lagrangian
            );
            println!("Iteration {} - residual: {}", iteration, residual);
        }
        old_imgs_hat = imgs_hat;
        if residual < config.threshold {
            break;
        }
    }

    // TODO: write singular values to a file.
    let final_imgs_registered = (0..nb_imgs)
        .map(|i| {
            crate::utils::reshape(
                imgs_registered
                    .column(i)
                    .map(|x| (config.image_max * x).min(255.0) as u8),
                height,
                width,
            )
        })
        .collect();
    Ok((final_imgs_registered, motion_vec))
}

/// Computes the sqrt of the sum of squared values.
/// This is the L2 norm of the vectorized version of the matrix.
fn norm(matrix: &DMatrix<f32>) -> f32 {
    norm_sqr(matrix).sqrt() as f32
}

fn norm_sqr(matrix: &DMatrix<f32>) -> f64 {
    matrix.iter().map(|&x| (x as f64).powi(2)).sum()
}

fn mat_from_vec<T1: Scalar, T2: Scalar, F: Fn(&T1) -> T2>(
    rows: usize,
    columns: usize,
    convert: &F,
    imgs: &[DMatrix<T1>],
) -> DMatrix<T2> {
    let nb_imgs = imgs.len();
    assert!(nb_imgs > 0);
    DMatrix::from_iterator(
        rows * columns,
        nb_imgs,
        imgs.iter().flat_map(|img| img.iter().map(convert)),
    )
}

fn shrink<T: RealField>(alpha: T, x: T) -> T {
    let alpha = alpha.abs();
    if x.is_sign_positive() {
        (x - alpha).max(T::zero())
    } else {
        (x + alpha).min(T::zero())
    }
}
