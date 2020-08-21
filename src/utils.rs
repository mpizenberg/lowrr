use nalgebra::base::dimension::{Dim, Dynamic};
use nalgebra::base::{Scalar, VecStorage};
use nalgebra::{DMatrix, Matrix};

pub fn rgb_to_gray(red: &DMatrix<u8>, green: &DMatrix<u8>, blue: &DMatrix<u8>) -> DMatrix<u8> {
    let (rows, cols) = red.shape();
    DMatrix::from_iterator(
        rows,
        cols,
        red.iter()
            .zip(green.iter())
            .zip(blue.iter())
            .map(|((&r, &g), &b)| {
                (0.2989 * r as f32 + 0.5870 * g as f32 + 0.1140 * b as f32).max(255.0) as u8
            }),
    )
}

/// Reshapes `self` in-place such that it has dimensions `new_nrows × new_ncols`.
///
/// The values are not copied or moved. This function will panic if dynamic sizes are provided
/// and not compatible.
pub fn reshape<N, R, C>(
    matrix: Matrix<N, R, C, VecStorage<N, R, C>>,
    nrows: usize,
    ncols: usize,
) -> DMatrix<N>
where
    N: Scalar,
    R: Dim,
    C: Dim,
{
    assert_eq!(nrows * ncols, matrix.data.len());
    let new_data = VecStorage::new(Dynamic::new(nrows), Dynamic::new(ncols), matrix.data.into());
    DMatrix::from_data(new_data)
}
