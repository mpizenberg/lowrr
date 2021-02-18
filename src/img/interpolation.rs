// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions to interpolate / extrapolate warped images.

use nalgebra::{DMatrix, Scalar, Vector3};
use std::ops::{Add, Mul};

/// Trait for types that can be linearly interpolated with the `linear` function.
///
/// The `Vector` generic type refers to the intermediate type used during interpolations.
/// It usually is the `f32` scalar or a vector of `f32` values.
///
/// The `Output` type is the final type obtained after interpolation.
/// It is obtained via conversion from the interpolated vector.
pub trait CanLinearInterpolate<Vector, Output>
where
    Vector: Add<Output = Vector>,
    f32: Mul<Vector, Output = Vector>,
{
    fn to_vector(self) -> Vector;
    fn from_vector(v: Vector) -> Output;
}

/// Implement CanLinearInterpolate for u8 with f32 outputs.
/// WARNING: beware that interpolating with a f32 output normalizes values from [0-255] to [0.0, 1.0].
impl CanLinearInterpolate<f32, f32> for u8 {
    fn to_vector(self) -> f32 {
        self as f32
    }
    fn from_vector(v: f32) -> f32 {
        (v / 255.0).max(0.0).min(1.0)
    }
}

/// Implement CanLinearInterpolate for u8.
impl CanLinearInterpolate<f32, u8> for u8 {
    fn to_vector(self) -> f32 {
        self as f32
    }
    fn from_vector(v: f32) -> u8 {
        v.max(0.0).min(u8::MAX as f32).round() as u8
    }
}

/// Implement CanLinearInterpolate for u16 with f32 outputs.
/// WARNING: beware that interpolating with a f32 output normalizes values from [0-65535] to [0.0, 1.0].
impl CanLinearInterpolate<f32, f32> for u16 {
    fn to_vector(self) -> f32 {
        self as f32
    }
    fn from_vector(v: f32) -> f32 {
        (v / u16::MAX as f32).max(0.0).min(1.0)
    }
}

/// Implement CanLinearInterpolate for u16.
impl CanLinearInterpolate<f32, u16> for u16 {
    fn to_vector(self) -> f32 {
        self as f32
    }
    fn from_vector(v: f32) -> u16 {
        v.max(0.0).min(u16::MAX as f32).round() as u16
    }
}

/// Implement CanLinearInterpolate for (T,T,T) if T also implements it.
impl<O, T: CanLinearInterpolate<f32, O>> CanLinearInterpolate<Vector3<f32>, (O, O, O)>
    for (T, T, T)
{
    fn to_vector(self) -> Vector3<f32> {
        Vector3::new(self.0.to_vector(), self.1.to_vector(), self.2.to_vector())
    }
    fn from_vector(v: Vector3<f32>) -> (O, O, O) {
        (
            T::from_vector(v.x),
            T::from_vector(v.y),
            T::from_vector(v.z),
        )
    }
}

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Extrapolate with the nearest border if the point is outside of the image boundaries.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn linear<V, O, T>(x: f32, y: f32, image: &DMatrix<T>) -> O
where
    V: Add<Output = V>,
    f32: Mul<V, Output = V>,
    T: Scalar + Copy + CanLinearInterpolate<V, O>,
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
        let a = x - u;
        let b = y - v;
        let vu_00 = image[(v_0, u_0)].to_vector();
        let vu_10 = image[(v_1, u_0)].to_vector();
        let vu_01 = image[(v_0, u_1)].to_vector();
        let vu_11 = image[(v_1, u_1)].to_vector();
        let interp = Mul::<f32>::mul(1.0 - b, 1.0 - a) * vu_00
            + Mul::<f32>::mul(b, 1.0 - a) * vu_10
            + Mul::<f32>::mul(1.0 - b, a) * vu_01
            + Mul::<f32>::mul(b, a) * vu_11;
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
