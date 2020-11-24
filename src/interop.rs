// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Interoperability conversions between the image and matrix types.

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use nalgebra::DMatrix;

/// Convert an `u8` matrix into a `GrayImage`.
/// Inverse operation of `matrix_from_image`.
///
/// Performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

/// Convert an `(u8,u8,8)` matrix into an `RgbImage`.
///
/// Performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn rgb_from_matrix(mat: &DMatrix<(u8, u8, u8)>) -> RgbImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = RgbImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let (r, g, b) = mat[(y as usize, x as usize)];
        *pixel = Rgb([r, g, b]);
    }
    img_buf
}

/// Create a gray image with a borrowed reference to the matrix buffer.
///
/// Very performant since no copy is performed,
/// but produces a transposed image due to differences in row/column major.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix_transposed(mat: &DMatrix<u8>) -> ImageBuffer<Luma<u8>, &[u8]> {
    let (nb_rows, nb_cols) = mat.shape();
    ImageBuffer::from_raw(nb_rows as u32, nb_cols as u32, mat.as_slice())
        .expect("Buffer not large enough")
}

/// Convert a `GrayImage` into an `u8` matrix.
/// Inverse operation of `image_from_matrix`.
pub fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}

/// Convert a `RgbImage` into an `(u8, u8, u8)` matrix.
/// Inverse operation of `rgb_from_matrix`.
pub fn matrix_from_rgb_image(img: RgbImage) -> DMatrix<(u8, u8, u8)> {
    let (width, height) = img.dimensions();
    // TODO: improve the suboptimal allocation in addition to transposition.
    DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw().chunks_exact(3).map(|s| (s[0], s[1], s[2])),
    )
    .transpose()
}

/// Convert a `RgbImage` into an `u8` matrix with green channel.
pub fn green_mat_from_rgb_image(img: RgbImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();

    // DMatrix::from_fn(height as usize, width as usize, |i, j| {
    //     img.get_pixel(j as u32, i as u32)[1]
    // })

    let mut mat = DMatrix::zeros(height as usize, width as usize);
    img.enumerate_pixels()
        .for_each(|(x, y, p)| *mat.get_mut((y as usize, x as usize)).unwrap() = p[1]);
    mat

    // DMatrix::from_iterator(height as usize, width as usize, img.pixels().map(|p| p[1]))
}
