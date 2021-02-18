// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! # Image manipulation
//!
//! This module is a namespace for submodules dealing with image manipulation.
//! The underlying data is almost always considered to be a 2D nalgebra matrix.

pub mod filter;
pub mod gradients;
pub mod interpolation;
pub mod multires;
pub mod registration;
pub mod viz;
