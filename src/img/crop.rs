use nalgebra::{DMatrix, Scalar, Vector6};
use std::str::FromStr;

#[derive(Debug)]
pub struct Crop {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

impl FromStr for Crop {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.splitn(4, ',').collect();
        assert!(parts.len() == 4,   "--crop argument must be of the shape left,top,right,bottom with no space between coordinates"
        );
        let left = parts[0].parse()?;
        let top = parts[1].parse()?;
        let right = parts[2].parse()?;
        let bottom = parts[3].parse()?;
        Ok(Crop {
            left,
            top,
            right,
            bottom,
        })
    }
}

pub fn crop<T: Scalar>(frame: &Crop, img: &DMatrix<T>) -> DMatrix<T> {
    let Crop {
        left,
        top,
        right,
        bottom,
    } = frame;
    let (height, width) = img.shape();
    assert!(left < &width, "Error: left >= image width");
    assert!(right < &width, "Error: right >= image width");
    assert!(top < &height, "Error: top >= image height");
    assert!(bottom < &height, "Error: bottom >= image height");
    assert!(left <= right, "Error: right must be greater than left");
    assert!(top <= bottom, "Error: bottom must be greater than top");
    let nrows = bottom - top;
    let ncols = right - left;
    img.slice((*top, *left), (nrows, ncols)).into_owned()
}

pub fn recover_original_motion(crop: &Crop, motion_vec_crop: &[Vector6<f32>]) -> Vec<Vector6<f32>> {
    let Crop { left, top, .. } = crop;
    let translation = crate::affine2d::projection_mat(&Vector6::new(
        0.0,
        0.0,
        0.0,
        0.0,
        *left as f32,
        *top as f32,
    ));
    let translation_inv = translation.try_inverse().unwrap();
    motion_vec_crop
        .iter()
        .map(|m| {
            let motion = crate::affine2d::projection_mat(m);
            crate::affine2d::projection_params(&(translation * motion * translation_inv))
        })
        .collect()
}
