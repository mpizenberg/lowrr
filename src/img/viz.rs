// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper module for visualizations.

use nalgebra::{DMatrix, Scalar};

pub trait ToRgb8 {
    fn to_rgb8(self) -> (u8, u8, u8);
}

impl ToRgb8 for u8 {
    fn to_rgb8(self) -> (u8, u8, u8) {
        (self, self, self)
    }
}

impl ToRgb8 for u16 {
    fn to_rgb8(self) -> (u8, u8, u8) {
        ((self / 256) as u8, (self / 256) as u8, (self / 256) as u8)
    }
}

pub fn mask_overlay<T: Scalar + ToRgb8>(
    mask: &DMatrix<bool>,
    img_mat: &DMatrix<T>,
) -> DMatrix<(u8, u8, u8)> {
    mask.zip_map(img_mat, |in_mask, pixel| {
        if in_mask {
            (255, 0, 0)
        } else {
            pixel.to_rgb8()
        }
    })
}
