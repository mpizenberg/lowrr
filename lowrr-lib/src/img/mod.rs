// SPDX-License-Identifier: MPL-2.0

//! # Image manipulation
//!
//! This module is a namespace for submodules dealing with image manipulation.
//! The underlying data is almost always considered to be a 2D nalgebra matrix.

pub mod crop;
pub mod filter;
pub mod gradients;
pub mod interpolation;
pub mod multires;
pub mod registration;
pub mod sparse;
pub mod viz;
