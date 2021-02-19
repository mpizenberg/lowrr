use nalgebra::{Matrix3, Vector6};

#[rustfmt::skip]
pub fn projection_mat(params: &Vector6<f32>) -> Matrix3<f32> {
    Matrix3::new(
        1.0 + params[0], params[2], params[4],
        params[1], 1.0 + params[3], params[5],
        0.0, 0.0, 1.0,
    )
}

pub fn projection_params(mat: &Matrix3<f32>) -> Vector6<f32> {
    Vector6::new(
        mat.m11 - 1.0,
        mat.m21,
        mat.m12,
        mat.m22 - 1.0,
        mat.m13,
        mat.m23,
    )
}
