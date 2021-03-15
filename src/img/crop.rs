// SPDX-License-Identifier: MPL-2.0

use nalgebra::{DMatrix, Scalar, Vector6};
use std::convert::TryFrom;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct Crop {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

#[derive(Error, Debug)]
pub enum CropError {
    #[error("Invalid crop frame coordinates: {0}")]
    InvalidFrame(String),
    #[error("Not enough arguments: expected 4 but got only {0}")]
    NotEnoughArgs(usize),
    #[error("Too many arguments: expected 4 but got more")]
    TooManyArgs,
    #[error("Error parsing crop frame coordinates")]
    Parse(#[from] std::num::ParseIntError),
}

impl TryFrom<clap::Values<'_>> for Crop {
    type Error = CropError;
    fn try_from(mut vs: clap::Values) -> Result<Self, Self::Error> {
        match (vs.next(), vs.next(), vs.next(), vs.next(), vs.next()) {
            (None, _, _, _, _) => Err(CropError::NotEnoughArgs(0)),
            (_, None, _, _, _) => Err(CropError::NotEnoughArgs(1)),
            (_, _, None, _, _) => Err(CropError::NotEnoughArgs(2)),
            (_, _, _, None, _) => Err(CropError::NotEnoughArgs(3)),
            (_, _, _, _, Some(_)) => Err(CropError::TooManyArgs),
            (Some(left), Some(top), Some(right), Some(bottom), None) => Ok(Crop {
                left: left.parse()?,
                top: top.parse()?,
                right: right.parse()?,
                bottom: bottom.parse()?,
            }),
        }
    }
}

pub fn crop<T: Scalar>(frame: Crop, img: &DMatrix<T>) -> Result<DMatrix<T>, CropError> {
    let Crop {
        left,
        top,
        right,
        bottom,
    } = frame;
    let (height, width) = img.shape();

    // Check that the frame coordinates make sense.
    if left >= width {
        return Err(CropError::InvalidFrame(format!(
            "left >= width ({} >= {})",
            left, width
        )));
    }
    if right > width {
        return Err(CropError::InvalidFrame(format!(
            "right > width ({} > {})",
            right, width
        )));
    }
    if top >= height {
        return Err(CropError::InvalidFrame(format!(
            "top >= height ({} >= {})",
            top, height
        )));
    }
    if bottom > height {
        return Err(CropError::InvalidFrame(format!(
            "bottom > height ({} > {})",
            bottom, height
        )));
    }
    if left >= right {
        return Err(CropError::InvalidFrame(format!(
            "left >= right ({} >= {})",
            left, right
        )));
    }
    if top >= bottom {
        return Err(CropError::InvalidFrame(format!(
            "top >= bottom ({} >= {})",
            top, bottom
        )));
    }
    // Extract the cropped slice.
    let nrows = bottom - top;
    let ncols = right - left;
    Ok(img.slice((top, left), (nrows, ncols)).into_owned())
}

pub fn recover_original_motion(crop: Crop, motion_vec_crop: &[Vector6<f32>]) -> Vec<Vector6<f32>> {
    let Crop { left, top, .. } = crop;
    let translation =
        crate::affine2d::projection_mat(&Vector6::new(0.0, 0.0, 0.0, 0.0, left as f32, top as f32));
    let translation_inv = translation.try_inverse().unwrap();
    motion_vec_crop
        .iter()
        .map(|m| {
            let motion = crate::affine2d::projection_mat(m);
            crate::affine2d::projection_params(&(translation * motion * translation_inv))
        })
        .collect()
}
