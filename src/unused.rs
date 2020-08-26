// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Mostly unused functions that were still fun
//! to write correctly with generics.

use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::Dim;
use nalgebra::base::Scalar;
use nalgebra::MatrixMN;

// Not really useful
// map_mut(&mut svd.singular_values, |x| {
//     *x = shrink(1.0 / config.rho, *x)
// });
pub fn map_mut<N: Scalar, R: Dim, C: Dim, F>(matrix: &mut MatrixMN<N, R, C>, f: F)
where
    DefaultAllocator: Allocator<N, R, C>,
    F: Fn(&mut N),
{
    for element in matrix.iter_mut() {
        f(element)
    }
}
