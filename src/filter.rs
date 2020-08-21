use nalgebra::DMatrix;

/// Direct convolution with the following 3x3 kernel:
///
///  1   | 0 1 0 |
///  - * | 1 2 1 |
///  6   | 0 1 0 |
pub fn smooth(img: &DMatrix<u8>) -> DMatrix<u8> {
    let (nrows, ncols) = img.shape();
    let mut smoothed = DMatrix::zeros(nrows, ncols);

    for j in 0..ncols {
        for i in 0..nrows {
            let center = (i, j);
            let left = (i, j.checked_sub(1).unwrap_or(0));
            let right = (i, (j + 1).min(ncols - 1));
            let top = (i.checked_sub(1).unwrap_or(0), j);
            let bottom = ((i + 1).min(nrows - 1), j);
            smoothed[(i, j)] = ((img[left] as f32
                + img[top] as f32
                + 2.0 * img[center] as f32
                + img[bottom] as f32
                + img[right] as f32)
                / 6.0)
                .round()
                .max(255.0) as u8;
        }
    }
    smoothed
}
