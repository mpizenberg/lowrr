// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions to interpolate / extrapolate warped images.

use nalgebra::base::{DMatrix, Scalar};

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Extrapolate if the point is outside of the image boundaries.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn linear<T>(x: f32, y: f32, image: &DMatrix<T>) -> f32
where
    T: Scalar + Copy + Into<f32>,
{
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as f32 && v >= 0.0 && v < (height - 2) as f32 {
        // Linear interpolation inside boundaries.
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let vu_00: f32 = image[(v_0, u_0)].into();
        let vu_10: f32 = image[(v_1, u_0)].into();
        let vu_01: f32 = image[(v_0, u_1)].into();
        let vu_11: f32 = image[(v_1, u_1)].into();
        let a = x - u;
        let b = y - v;
        (1.0 - b) * (1.0 - a) * vu_00
            + b * (1.0 - a) * vu_10
            + (1.0 - b) * a * vu_01
            + b * a * vu_11
    } else {
        // Nearest neighbour extrapolation outside boundaries.
        image[nearest_border(x, y, width, height)].into()
    }
}

fn nearest_border(x: f32, y: f32, width: usize, height: usize) -> (usize, usize) {
    let u = x.max(0.0).min((width - 1) as f32) as usize;
    let v = y.max(0.0).min((height - 1) as f32) as usize;
    (v, u)
}
