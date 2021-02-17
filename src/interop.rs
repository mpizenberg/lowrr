// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Interoperability conversions between the image and matrix types.

use image::{ImageBuffer, Luma, Primitive, Rgb};
use nalgebra::{DMatrix, Scalar};

/// Convert a matrix into a gray level image.
/// Inverse operation of `matrix_from_image`.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix<T: Scalar + Primitive>(mat: &DMatrix<T>) -> ImageBuffer<Luma<T>, Vec<T>> {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

/// Convert a `(T,T,T)` RGB matrix into an RGB image.
/// Inverse operation of matrix_from_rgb_image.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn rgb_from_matrix<T: Scalar + Primitive>(
    mat: &DMatrix<(T, T, T)>,
) -> ImageBuffer<Rgb<T>, Vec<T>> {
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let (r, g, b) = mat[(y as usize, x as usize)];
        *pixel = Rgb([r, g, b]);
    }
    img_buf
}

/// Convert a gray image into a matrix.
/// Inverse operation of `image_from_matrix`.
pub fn matrix_from_image<T: Scalar + Primitive>(img: ImageBuffer<Luma<T>, Vec<T>>) -> DMatrix<T> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}

/// Convert an RGB image into a `(T, T, T)` RGB matrix.
/// Inverse operation of `rgb_from_matrix`.
pub fn matrix_from_rgb_image<T: Scalar + Primitive>(
    img: ImageBuffer<Rgb<T>, Vec<T>>,
) -> DMatrix<(T, T, T)> {
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (width, height) = img.dimensions();
    DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw().chunks_exact(3).map(|s| (s[0], s[1], s[2])),
    )
    .transpose()
}
