// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper module for functions that didn't fit anywhere else.

use nalgebra::base::dimension::{Dim, Dynamic};
use nalgebra::base::{Scalar, VecStorage};
use nalgebra::{DMatrix, Matrix};
use std::ops::Mul;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::interop::IntoImage;

#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("Failed to create directory {dir} with the following error: {source}")]
    CreateDir {
        dir: PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to save image {path} with the following error: {source}")]
    SavingImg {
        path: PathBuf,
        source: image::ImageError,
    },
}

/// Same as rgb2gray matlab function, but for u8.
pub fn rgb_to_gray(red: &DMatrix<u8>, green: &DMatrix<u8>, blue: &DMatrix<u8>) -> DMatrix<u8> {
    let (rows, cols) = red.shape();
    DMatrix::from_iterator(
        rows,
        cols,
        red.iter()
            .zip(green.iter())
            .zip(blue.iter())
            .map(|((&r, &g), &b)| {
                (0.2989 * r as f32 + 0.5870 * g as f32 + 0.1140 * b as f32).max(255.0) as u8
            }),
    )
}

/// Reshapes `self` in-place such that it has dimensions `nrows × ncols`.
///
/// The values are not copied or moved. This function will panic if
/// provided dynamic sizes are not compatible.
pub fn reshape<N, R, C>(
    matrix: Matrix<N, R, C, VecStorage<N, R, C>>,
    nrows: usize,
    ncols: usize,
) -> DMatrix<N>
where
    N: Scalar,
    R: Dim,
    C: Dim,
{
    assert_eq!(nrows * ncols, matrix.data.len());
    let new_data = VecStorage::new(Dynamic::new(nrows), Dynamic::new(ncols), matrix.data.into());
    DMatrix::from_data(new_data)
}

/// Transpose a Vec of Vec.
/// Will panic if the inner vecs are not all the same size.
pub fn transpose<T: Clone>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    // Checking case of an empty vec.
    if v.is_empty() {
        return Vec::new();
    }

    // Checking case of vec of empty vec.
    let transposed_len = v[0].len();
    assert!(v.iter().all(|vi| vi.len() == transposed_len));
    if transposed_len == 0 {
        return Vec::new();
    }

    // Normal case.
    let mut v_transposed = vec![Vec::new(); transposed_len];
    for vi in v.into_iter() {
        for (v_tj, vj) in v_transposed.iter_mut().zip(vi.into_iter()) {
            v_tj.push(vj);
        }
    }
    v_transposed
}

/// Save a bunch of images into the given directory.
pub fn save_all_imgs<P: AsRef<Path>, I: IntoImage>(dir: P, imgs: &[I]) -> Result<(), UtilsError> {
    let pb = indicatif::ProgressBar::new(imgs.len() as u64);
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|source| UtilsError::CreateDir {
        dir: PathBuf::from(dir),
        source,
    })?;
    for (i, img) in imgs.iter().enumerate() {
        let img_path = dir.join(format!("{}.png", i));
        img.into_image()
            .save(&img_path)
            .map_err(|source| UtilsError::SavingImg {
                path: img_path,
                source,
            })?;
        pb.inc(1);
    }
    pb.finish();
    Ok(())
}

// Helper functions to play with coordinates iterators.

/// Retrieve the coordinates of selected pixels in a binary mask.
pub fn coordinates_from_mask(mask: &DMatrix<bool>) -> Vec<(usize, usize)> {
    crate::img::sparse::extract(mask.iter().cloned(), coords_col_major(mask.shape())).collect()
}

/// An iterator over all coordinates of a matrix in column major.
pub fn coords_col_major(shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let (height, width) = shape;
    let coords = (0..width).map(move |x| (0..height).map(move |y| (x, y)));
    coords.flatten()
}

/// An iterator over all coordinates of a matrix in row major.
pub fn coords_row_major(shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let (height, width) = shape;
    let coords = (0..height).map(move |y| (0..width).map(move |x| (x, y)));
    coords.flatten()
}

// Helper functions to equalize the mean intensity of a collection of images.

/// Only work for gray images for now.
pub trait CanEqualize: Scalar + Copy + Mul + Into<f32> {
    /// Convert the target, set as a float in [0,1], into an equivalent value for current type.
    fn target_mean(target: f32) -> f32;
    /// Convert the scaled f32 value back into the current type.
    fn from_as(f: f32) -> Self;
}

impl CanEqualize for u8 {
    fn target_mean(target: f32) -> f32 {
        u8::MAX as f32 * target
    }
    fn from_as(f: f32) -> Self {
        f.max(0.0).min(255.0) as Self
    }
}

impl CanEqualize for u16 {
    fn target_mean(target: f32) -> f32 {
        u16::MAX as f32 * target
    }
    fn from_as(f: f32) -> Self {
        f.max(0.0).min(65535.0) as Self
    }
}

impl CanEqualize for f32 {
    fn target_mean(target: f32) -> f32 {
        target
    }
    fn from_as(f: f32) -> Self {
        f
    }
}

/// Change the mean intensity of all images to be approximately the same.
pub fn equalize_mean<T: CanEqualize>(target: f32, imgs: &mut [DMatrix<T>]) {
    // Compute mean intensities.
    let sum_intensities: Vec<f32> = imgs
        .iter()
        .map(|im| im.iter().map(|x| (*x).into()).sum())
        .collect();
    let nb_pixels = imgs[0].len();
    let mean_intensities: Vec<_> = sum_intensities
        .iter()
        .map(|x| x / nb_pixels as f32)
        .collect();
    log::info!("mean intensities {:?}", mean_intensities);

    // Multiply all images such that the mean intensity is near the target.
    for (im, mean) in imgs.iter_mut().zip(mean_intensities) {
        let scale = T::target_mean(target) / mean;
        for pixel in im.iter_mut() {
            *pixel = T::from_as(Mul::<f32>::mul(scale, (*pixel).into()));
        }
    }
}
