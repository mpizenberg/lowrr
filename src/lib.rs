// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! # Low rank registration
//!
//! Low-rank registration of slightly unaligned images for photometric stereo.

// #![warn(missing_docs)]

pub mod filter;
pub mod gradients;
pub mod interop;
pub mod interpolation;
pub mod multires;
pub mod optimizer;
pub mod registration;
pub mod utils;
