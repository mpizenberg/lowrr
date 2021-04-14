// SPDX-License-Identifier: MPL-2.0

use image::GenericImageView;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

#[macro_use]
mod utils; // define console_log! macro

#[wasm_bindgen]
pub fn crop(file: &[u8]) -> Box<[u8]> {
    utils::set_panic_hook();
    console_log!("cropping!");
    let reader = image::io::Reader::new(Cursor::new(file))
        .with_guessed_format()
        .expect("Cursor io never fails");
    let image = reader.decode().expect("Error decoding the image");
    let new_width = image.width() / 2;
    let new_height = image.height() / 2;
    let cropped_image = image.crop_imm(0, 0, new_width, new_height);
    let mut cropped_buffer: Vec<u8> = Vec::new();
    cropped_image
        .write_to(&mut cropped_buffer, image::ImageOutputFormat::Png)
        .expect("Error encoding the cropped image to PNG");
    cropped_buffer.into_boxed_slice()
}
