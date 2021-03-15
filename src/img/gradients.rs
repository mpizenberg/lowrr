// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper function to compute gradients

use nalgebra::{DMatrix, Scalar};
use std::ops::{Add, Div, Mul, Sub};

/// Compute a centered gradient.
///
/// 1/2 * ( img(i+1,j) - img(i-1,j), img(i,j+1) - img(i,j-1) )
///
/// Gradients of pixels at the border of the image are set to 0.
#[allow(clippy::similar_names)]
pub fn centered(img: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
    // TODO: might be better to return DMatrix<(i16,i16)>?
    let (nb_rows, nb_cols) = img.shape();
    assert!(
        nb_rows > 2,
        "Impossible to compute centered gradients with only {} rows",
        nb_rows
    );
    assert!(
        nb_cols > 2,
        "Impossible to compute centered gradients with only {} columns",
        nb_cols
    );
    let top = img.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = img.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = img.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = img.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut grad_x = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_y = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_x_inner = grad_x.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    let mut grad_y_inner = grad_y.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            grad_x_inner[(i, j)] = (i16::from(right[(i, j)]) - i16::from(left[(i, j)])) / 2;
            grad_y_inner[(i, j)] = (i16::from(bottom[(i, j)]) - i16::from(top[(i, j)])) / 2;
        }
    }
    (grad_x, grad_y)
}

/// Compute a centered gradient.
///
/// 1/2 * ( img(i+1,j) - img(i-1,j), img(i,j+1) - img(i,j-1) )
///
/// Gradients of pixels at the border of the image are set to 0.
#[allow(clippy::similar_names)]
pub fn centered_f32(img: &DMatrix<f32>) -> DMatrix<(f32, f32)> {
    // TODO: might be better to return DMatrix<(i16,i16)>?
    let (nb_rows, nb_cols) = img.shape();
    assert!(
        nb_rows > 2,
        "Impossible to compute centered gradients with only {} rows",
        nb_rows
    );
    assert!(
        nb_cols > 2,
        "Impossible to compute centered gradients with only {} columns",
        nb_cols
    );
    let top = img.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = img.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = img.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = img.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut grad = DMatrix::repeat(nb_rows, nb_cols, (0.0, 0.0));
    let mut grad_inner = grad.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            let gx = 0.5 * (right[(i, j)] - left[(i, j)]);
            let gy = 0.5 * (bottom[(i, j)] - top[(i, j)]);
            grad_inner[(i, j)] = (gx, gy);
        }
    }
    grad
}

/// Compute a centered gradient of 4th order.
///
/// The coefficients are 1/12 * [ 1  -8  8  -1 ]
///
/// Gradients of pixels at the border of the image are set to 0.
#[allow(clippy::similar_names)]
pub fn centered_4(img: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
    let (nb_rows, nb_cols) = img.shape();
    assert!(
        nb_rows > 4,
        "Impossible to compute centered gradients with only {} rows",
        nb_rows
    );
    assert!(
        nb_cols > 4,
        "Impossible to compute centered gradients with only {} columns",
        nb_cols
    );

    let img_i16 = img.map(|x| x as i16);

    let left_2 = img_i16.slice((2, 0), (nb_rows - 4, nb_cols - 4));
    let left_1 = img_i16.slice((2, 1), (nb_rows - 4, nb_cols - 4));
    let right_1 = img_i16.slice((2, 2), (nb_rows - 4, nb_cols - 4));
    let right_2 = img_i16.slice((2, 3), (nb_rows - 4, nb_cols - 4));

    let top_2 = img_i16.slice((0, 2), (nb_rows - 4, nb_cols - 4));
    let top_1 = img_i16.slice((1, 2), (nb_rows - 4, nb_cols - 4));
    let bottom_1 = img_i16.slice((2, 2), (nb_rows - 4, nb_cols - 4));
    let bottom_2 = img_i16.slice((3, 2), (nb_rows - 4, nb_cols - 4));

    let mut grad_x = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_y = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_x_inner = grad_x.slice_mut((2, 2), (nb_rows - 4, nb_cols - 4));
    let mut grad_y_inner = grad_y.slice_mut((2, 2), (nb_rows - 4, nb_cols - 4));

    // TODO: clearly not memory efficient regarding allocations...
    //
    // One solution would be to decompose each operation with .axpy()
    // Another would be to use iterators.
    // Both are very unreadable.
    grad_x_inner.copy_from(&((left_2 - 8 * left_1 + 8 * right_1 - right_2) / 12));
    grad_y_inner.copy_from(&((top_2 - 8 * top_1 + 8 * bottom_1 - bottom_2) / 12));

    (grad_x, grad_y)
}

/// Compute a centered gradient of 4th order (in f32).
///
/// The coefficients are 1/12 * [ 1  -8  8  -1 ]
///
/// Gradients of pixels at the border of the image are set to 0.
#[allow(clippy::similar_names)]
pub fn centered_4_f32(img: &DMatrix<f32>) -> (DMatrix<f32>, DMatrix<f32>) {
    let (nb_rows, nb_cols) = img.shape();
    assert!(
        nb_rows > 4,
        "Impossible to compute centered gradients with only {} rows",
        nb_rows
    );
    assert!(
        nb_cols > 4,
        "Impossible to compute centered gradients with only {} columns",
        nb_cols
    );

    let left_2 = img.slice((2, 0), (nb_rows - 4, nb_cols - 4));
    let left_1 = img.slice((2, 1), (nb_rows - 4, nb_cols - 4));
    let right_1 = img.slice((2, 3), (nb_rows - 4, nb_cols - 4));
    let right_2 = img.slice((2, 4), (nb_rows - 4, nb_cols - 4));

    let top_2 = img.slice((0, 2), (nb_rows - 4, nb_cols - 4));
    let top_1 = img.slice((1, 2), (nb_rows - 4, nb_cols - 4));
    let bottom_1 = img.slice((3, 2), (nb_rows - 4, nb_cols - 4));
    let bottom_2 = img.slice((4, 2), (nb_rows - 4, nb_cols - 4));

    let mut grad_x = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_y = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_x_inner = grad_x.slice_mut((2, 2), (nb_rows - 4, nb_cols - 4));
    let mut grad_y_inner = grad_y.slice_mut((2, 2), (nb_rows - 4, nb_cols - 4));

    // TODO: clearly not memory efficient regarding allocations...
    //
    // One solution would be to decompose each operation with .axpy()
    // Another would be to use iterators.
    // Both are very unreadable.
    grad_x_inner.copy_from(&((left_2 - 8.0 * left_1 + 8.0 * right_1 - right_2) / 12.0));
    grad_y_inner.copy_from(&((top_2 - 8.0 * top_1 + 8.0 * bottom_1 - bottom_2) / 12.0));

    (grad_x, grad_y)
}

/// Compute squared gradient norm from x and y gradient matrices.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn squared_norm(grad_x: &DMatrix<i16>, grad_y: &DMatrix<i16>) -> DMatrix<u16> {
    grad_x.zip_map(grad_y, |gx, gy| {
        let gx = i32::from(gx);
        let gy = i32::from(gy);
        (gx * gx + gy * gy) as u16
    })
}

pub trait Bigger<B: Copy>: Copy {
    type BigSigned: Copy
        + Add<Output = Self::BigSigned>
        + Sub<Output = Self::BigSigned>
        + Mul<Output = Self::BigSigned>
        + Div<Output = Self::BigSigned>
        + From<u8>
        + From<Self>;
    fn from_as(b: Self::BigSigned) -> B;
    fn zero() -> B;
}

impl Bigger<u16> for u8 {
    type BigSigned = i32;
    fn from_as(b: Self::BigSigned) -> u16 {
        b as u16
    }
    fn zero() -> u16 {
        0
    }
}

impl Bigger<u32> for u16 {
    type BigSigned = i64;
    fn from_as(b: Self::BigSigned) -> u32 {
        b as u32
    }
    fn zero() -> u32 {
        0
    }
}

/// Compute squared gradient norm directly from the image.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn squared_norm_direct<T, U>(im: &DMatrix<T>) -> DMatrix<U>
where
    T: Scalar + Copy + Bigger<U>,
    U: Scalar + Copy,
{
    let (nb_rows, nb_cols) = im.shape();
    assert!(
        nb_rows > 2,
        "Impossible to compute gradients squared norms with only {} rows",
        nb_rows
    );
    assert!(
        nb_cols > 2,
        "Impossible to compute gradients squared norms with only {} columns",
        nb_cols
    );
    let top = im.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = im.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = im.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = im.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut squared_norm_mat = DMatrix::repeat(nb_rows, nb_cols, T::zero());
    let mut grad_inner = squared_norm_mat.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            let gx = T::BigSigned::from(right[(i, j)]) - T::BigSigned::from(left[(i, j)]);
            let gy = T::BigSigned::from(bottom[(i, j)]) - T::BigSigned::from(top[(i, j)]);
            grad_inner[(i, j)] = T::from_as((gx * gx + gy * gy) / T::BigSigned::from(4));
        }
    }
    squared_norm_mat
}

// BLOCS 2x2 ###################################################################

/// Horizontal gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_x(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = i16::from(a);
    let b = i16::from(b);
    let c = i16::from(c);
    let d = i16::from(d);
    (c + d - a - b) / 2
}

/// Vertical gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_y(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = i16::from(a);
    let b = i16::from(b);
    let c = i16::from(c);
    let d = i16::from(d);
    (b - a + d - c) / 2
}

/// Gradient squared norm in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn bloc_squared_norm(a: u8, b: u8, c: u8, d: u8) -> u16 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);
    let d = i32::from(d);
    let dx = c + d - a - b;
    let dy = b - a + d - c;
    // I have checked that the max value is in u16.
    ((dx * dx + dy * dy) / 4) as u16
}
