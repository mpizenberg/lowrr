// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions to interpolate / extrapolate warped images.

use nalgebra::{DMatrix, Scalar, Vector3};
use std::ops::{Add, Mul};

/// Trait for types that can be linearly interpolated with the `linear` function.
pub trait LinearInterpolate<Float, Vector, Output>
where
    Vector: Add<Output = Vector>,
    Float: Mul<Vector, Output = Vector>,
{
    fn to_vector(self) -> Vector;
    fn from_vector(v: Vector) -> Output;
}

/// Implement LinearInterpolate for u8 RGB.
impl LinearInterpolate<f32, Vector3<f32>, (f32, f32, f32)> for (u8, u8, u8) {
    fn to_vector(self) -> Vector3<f32> {
        Vector3::new(self.0 as f32, self.1 as f32, self.2 as f32)
    }
    fn from_vector(v: Vector3<f32>) -> (f32, f32, f32) {
        (v.x, v.y, v.z)
    }
}

/// Implement LinearInterpolate for u8 gray.
impl LinearInterpolate<f32, f32, f32> for u8 {
    fn to_vector(self) -> f32 {
        self as f32
    }
    fn from_vector(v: f32) -> f32 {
        v
    }
}

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Extrapolate if the point is outside of the image boundaries.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn linear<
    V: Add<Output = V>,
    Float: From<f32> + Mul<V, Output = V>,
    O,
    T: Scalar + Copy + LinearInterpolate<Float, V, O>,
>(
    x: f32,
    y: f32,
    image: &DMatrix<T>,
) -> O {
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as f32 && v >= 0.0 && v < (height - 2) as f32 {
        // Linear interpolation inside boundaries.
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let a = x - u;
        let b = y - v;
        let vu_00 = image[(v_0, u_0)].to_vector();
        let vu_10 = image[(v_1, u_0)].to_vector();
        let vu_01 = image[(v_0, u_1)].to_vector();
        let vu_11 = image[(v_1, u_1)].to_vector();
        let interp = Float::from((1.0 - b) * (1.0 - a)) * vu_00
            + Float::from(b * (1.0 - a)) * vu_10
            + Float::from((1.0 - b) * a) * vu_01
            + Float::from(b * a) * vu_11;
        T::from_vector(interp)
    } else {
        // Nearest neighbour extrapolation outside boundaries.
        T::from_vector(image[nearest_border(x, y, width, height)].to_vector())
    }
}

fn nearest_border(x: f32, y: f32, width: usize, height: usize) -> (usize, usize) {
    let u = x.max(0.0).min((width - 1) as f32) as usize;
    let v = y.max(0.0).min((height - 1) as f32) as usize;
    (v, u)
}
