use nalgebra::DMatrix;

pub fn smooth(img: &DMatrix<u8>) -> DMatrix<u8> {
    let (rows, cols) = img.shape();
    let mut smoothed = DMatrix::<u8>::zeros(rows, cols);

    // img slices.
    let top = img.slice((0, 1), (rows - 2, cols - 2));
    let bottom = img.slice((2, 1), (rows - 2, cols - 2));
    let left = img.slice((1, 0), (rows - 2, cols - 2));
    let right = img.slice((1, 2), (rows - 2, cols - 2));
    let center = img.slice((1, 1), (rows - 2, cols - 2));

    // Update smoothed center.
    let mut smoothed_center = smoothed.slice_mut((1, 1), (rows - 2, cols - 2));
    for (i, x) in smoothed_center.iter_mut().enumerate() {
        *x = (left[i] + top[i] + 2 * center[i] + bottom[i] + right[i]) / 6;
    }
    unimplemented!()
}
