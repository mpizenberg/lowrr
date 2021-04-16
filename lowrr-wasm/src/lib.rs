// SPDX-License-Identifier: MPL-2.0

use image::{DynamicImage, GenericImageView};
use nalgebra::DMatrix;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use lowrr::img::crop::{crop, recover_original_motion, Crop};
use lowrr::img::registration::{self, CanRegister};
use lowrr::interop::{IntoDMatrix, ToImage};

#[macro_use]
mod utils; // define console_log! macro

#[wasm_bindgen]
pub struct Lowrr {
    image_ids: Vec<String>,
    dataset: Dataset,
    args: Option<Args>,
}

enum Dataset {
    Empty,
    GrayImages(Vec<DMatrix<u8>>),
    GrayImagesU16(Vec<DMatrix<u16>>),
    RgbImages(Vec<DMatrix<(u8, u8, u8)>>),
    RgbImagesU16(Vec<DMatrix<(u16, u16, u16)>>),
}

#[wasm_bindgen]
/// Type holding the algorithm parameters
pub struct Args {
    pub config: registration::Config,
    pub equalize: Option<f32>,
    pub crop: Option<Crop>,
}

#[wasm_bindgen]
impl Lowrr {
    pub fn init() -> Self {
        utils::set_panic_hook();
        Self {
            image_ids: Vec::new(),
            dataset: Dataset::Empty,
            args: None,
        }
    }

    pub fn load(&mut self, id: String, img_file: &[u8]) -> Result<(), JsValue> {
        console_log!("Loading an image");
        let reader = image::io::Reader::new(Cursor::new(img_file))
            .with_guessed_format()
            .expect("Cursor io never fails");
        // let image = reader.decode().expect("Error decoding the image");
        let dyn_img = reader.decode().map_err(|e| e.to_string())?;

        match (&dyn_img, &mut self.dataset) {
            // Loading the first image (empty dataset)
            (DynamicImage::ImageLuma8(_), Dataset::Empty) => {
                log::warn!("Images are of type Gray u8");
                self.dataset = Dataset::GrayImages(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageLuma8(_), Dataset::GrayImages(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageLuma16(_), Dataset::Empty) => {
                log::warn!("Images are of type Gray u16");
                self.dataset = Dataset::GrayImagesU16(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageLuma16(_), Dataset::GrayImagesU16(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageRgb8(_), Dataset::Empty) => {
                log::warn!("Images are of type RGB (u8, u8, u8)");
                self.dataset = Dataset::RgbImages(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageRgb8(_), Dataset::RgbImages(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageRgb16(_), Dataset::Empty) => {
                log::warn!("Images are of type RGB (u16, u16, u16)");
                self.dataset = Dataset::RgbImagesU16(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageRgb16(_), Dataset::RgbImagesU16(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            (DynamicImage::ImageBgr8(_), _) => Err("BGR order not supported")?,
            (DynamicImage::ImageBgra8(_), _) => Err("BGR order not supported")?,
            (DynamicImage::ImageLumaA8(_), _) => Err("Alpha channel not supported")?,
            (DynamicImage::ImageLumaA16(_), _) => Err("Alpha channel not supported")?,
            (DynamicImage::ImageRgba8(_), _) => Err("Alpha channel not supported")?,
            (DynamicImage::ImageRgba16(_), _) => Err("Alpha channel not supported")?,
            _ => Err("Images are not all of the same type")?,
        }

        Ok(())
    }

    pub fn run(args: Args) -> Result<(), JsValue> {
        // TODO: load dataset.
        Ok(())
    }
}
