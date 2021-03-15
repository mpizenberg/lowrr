// SPDX-License-Identifier: MPL-2.0

//! Helper functions for generation of multi-resolution data.

use nalgebra::{DMatrix, Scalar};
use std::ops::{Add, Div};

pub trait Bigger: Copy {
    type Big: Copy + Add<Output = Self::Big> + Div<Output = Self::Big> + From<u8> + From<Self>;
    fn from_as(b: Self::Big) -> Self;
}

impl Bigger for u8 {
    type Big = u16;
    #[allow(clippy::cast_possible_truncation)]
    fn from_as(b: Self::Big) -> Self {
        b as Self
    }
}

impl Bigger for u16 {
    type Big = u32;
    #[allow(clippy::cast_possible_truncation)]
    fn from_as(b: Self::Big) -> Self {
        b as Self
    }
}

/// Recursively generate a pyramid of matrices where each following level
/// is half the previous resolution, computed with the mean of each 2x2 block.
///
/// It consumes the original matrix to keep it as first level of the pyramid without copy.
/// If you need it later, simply use something like `pyramid[0]`.
///
/// PS: since we are using 2x2 blocs,
/// border information is lost for odd resolutions.
/// Some precision is also left to keep the pyramid data as `u8`.
pub fn mean_pyramid<T: Scalar + Copy + Bigger>(
    max_levels: usize,
    mat: DMatrix<T>,
) -> Vec<DMatrix<T>> {
    limited_sequence(max_levels, mat, |m| {
        halve(m, |a, b, c, d| {
            let a = T::Big::from(a);
            let b = T::Big::from(b);
            let c = T::Big::from(c);
            let d = T::Big::from(d);
            T::from_as((a + b + c + d) / T::Big::from(4u8))
        })
    })
}

/// Recursively apply a function transforming an image
/// until it's not possible anymore or the max length is reached.
///
/// Using `max_length = 0` has the same effect than `max_length = 1` since
/// the result vector contains always at least one matrix (the init matrix).
pub fn limited_sequence<F: Fn(&T) -> Option<T>, T>(max_length: usize, data: T, f: F) -> Vec<T> {
    let mut length = 1;
    let f_limited = |x: &T| {
        if length < max_length {
            length += 1;
            f(x)
        } else {
            None
        }
    };
    sequence(data, f_limited)
}

/// Recursively apply a function transforming data
/// until it's not possible anymore.
pub fn sequence<F: FnMut(&T) -> Option<T>, T>(data: T, mut f: F) -> Vec<T> {
    let mut s = Vec::new();
    s.push(data);
    while let Some(new_data) = f(s.last().unwrap()) {
        s.push(new_data);
    }
    s
}

/// Halve the resolution of a matrix by applying a function to each 2x2 block.
///
/// If one size of the matrix is < 2 then this function returns None.
/// If one size is odd, its last line/column is dropped.
#[allow(clippy::many_single_char_names)]
pub fn halve<F, T, U>(mat: &DMatrix<T>, f: F) -> Option<DMatrix<U>>
where
    F: Fn(T, T, T, T) -> U,
    T: Scalar + Copy,
    U: Scalar,
{
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        let half_mat = DMatrix::<U>::from_fn(half_r, half_c, |i, j| {
            let a = mat[(2 * i, 2 * j)];
            let b = mat[(2 * i + 1, 2 * j)];
            let c = mat[(2 * i, 2 * j + 1)];
            let d = mat[(2 * i + 1, 2 * j + 1)];
            f(a, b, c, d)
        });
        Some(half_mat)
    }
}

// Gradients stuff ###################################################

/// Compute centered gradients norm at each resolution from
/// the image at the higher resolution.
///
/// As a consequence there is one less level in the gradients pyramid.
/// The input matrices are given with decreasing resolution.
pub fn gradients_squared_norm(multires_mat: &[DMatrix<u8>]) -> Vec<DMatrix<u16>> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| {
            halve(mat, crate::img::gradients::bloc_squared_norm)
                .expect("gradients_squared_norm: error due to image with less than 2 pixels")
        })
        .collect()
}

/// Compute centered gradients at each resolution from
/// the image at the higher resolution.
///
/// As a consequence there is one less level in the gradients pyramid.
/// The input matrices are given with decreasing resolution.
pub fn gradients_xy(multires_mat: &[DMatrix<u8>]) -> Vec<(DMatrix<i16>, DMatrix<i16>)> {
    // TODO: maybe it would be better to return Vec<DMatrix<(i16,i16)>>,
    // to colocate the x and y gradient and do only one "halve" call?
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| {
            (
                halve(mat, crate::img::gradients::bloc_x)
                    .expect("gradients_xy x: error due to image with less than 2 pixels"),
                halve(mat, crate::img::gradients::bloc_y)
                    .expect("gradients_xy y: error due to image with less than 2 pixels"),
            )
        })
        .collect()
}
