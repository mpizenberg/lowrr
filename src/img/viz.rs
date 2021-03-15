// SPDX-License-Identifier: MPL-2.0

//! Helper module for visualizations.

use nalgebra::{DMatrix, Scalar};

/// Transform an RGB value into a single channel gray value.
pub trait IntoGray {
    type Output: Scalar;
    fn into_gray(self) -> Self::Output;
}

impl IntoGray for u8 {
    type Output = u8;
    fn into_gray(self) -> Self::Output {
        self
    }
}

impl IntoGray for u16 {
    type Output = u16;
    fn into_gray(self) -> Self::Output {
        self
    }
}

impl IntoGray for (u8, u8, u8) {
    type Output = u8;
    fn into_gray(self) -> Self::Output {
        let (_, g, _) = self;
        g
    }
}

impl IntoGray for (u16, u16, u16) {
    type Output = u16;
    fn into_gray(self) -> Self::Output {
        let (_, g, _) = self;
        g
    }
}

/// Ugrade a mono-channel value to a gray RGB value.
pub trait IntoRgb8 {
    fn into_rgb8(self) -> (u8, u8, u8);
}

impl IntoRgb8 for u8 {
    fn into_rgb8(self) -> (u8, u8, u8) {
        (self, self, self)
    }
}

impl IntoRgb8 for u16 {
    fn into_rgb8(self) -> (u8, u8, u8) {
        ((self / 256) as u8, (self / 256) as u8, (self / 256) as u8)
    }
}

pub fn mask_overlay<T: Scalar + IntoRgb8>(
    mask: &DMatrix<bool>,
    img_mat: &DMatrix<T>,
) -> DMatrix<(u8, u8, u8)> {
    mask.zip_map(img_mat, |in_mask, pixel| {
        if in_mask {
            (255, 0, 0)
        } else {
            pixel.into_rgb8()
        }
    })
}
