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
