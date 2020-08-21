use nalgebra::DMatrix;

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
