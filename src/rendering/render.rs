
use crate::rendering::shaders::RenderingShader;
use crate::rendering::gl::{get_model_view_matrix, get_projection_matrix, get_viewport_matrix};
use crate::rendering::drawing::draw_faces;

use std::sync::Arc;
use obj::{Obj, TexturedVertex};
use image::{ImageBuffer, RgbaImage, Rgba};
use nalgebra::{Point3, Vector3, Matrix2x3, Matrix3};

const WIDTH: u32 = 200;
const HEIGHT: u32 = 200;

/// Represents the camera properties used for rendering
struct Camera {
    pub position: Point3<f32>,
    pub focal_length: f32,
    pub view_point: Point3<f32>,
}

pub fn render(
    model: Arc<Obj<TexturedVertex>>,
    texture: ImageBuffer<Rgba<u8>, Vec<u8>>,
    normal_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
    specular_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> RgbaImage {

    // Create buffers for colour and z values
    let mut colour_buffer = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([0, 0, 0, 0]));
    let mut z_buffer = vec![vec![f32::NEG_INFINITY; HEIGHT as usize]; WIDTH as usize];

    // Create a shadow buffer for the shader
    let shadow_buffer = z_buffer.clone();

    // Configure camera and transformations
    let camera = Camera {
        position: Point3::new(0., 0., 1.0),
        focal_length: 3.0,
        view_point: Point3::origin(),
    };

    // Calculate the model-view, projection and viewport matrices
    let model_view = get_model_view_matrix(
        camera.position,
        camera.view_point,
        Point3::origin(),
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    let projection = get_projection_matrix(camera.focal_length);
    let viewport = get_viewport_matrix(HEIGHT as f32, WIDTH as f32, 1024.0);

    // Calculate inverse-transpose of the model-view matrix
    let model_view_it = model_view.try_inverse().unwrap().transpose();

    // Compute the shadow matrix (example: using directional light)
    let dir_light = Vector3::new(-1.0, 0.0, 1.5);
    let shadow_mv_mat = get_model_view_matrix(
        Point3::origin() + dir_light,
        Point3::origin(),
        Point3::origin(),
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
    );

    // Initialise the shader with the additional fields
    let mut shader = RenderingShader {
        model: &model,
        shadow_buffer: &shadow_buffer,
        uniform_model_view: model_view,
        uniform_model_view_it: model_view_it,
        uniform_shadow_mv_mat: shadow_mv_mat,
        uniform_projection: projection,
        uniform_viewport: viewport,
        uniform_texture: texture,
        uniform_normal_map: normal_map,
        uniform_specular_map: specular_map,
        uniform_ambient_light: 5.0,
        uniform_dir_light: dir_light.normalize(),
        varying_uv: Matrix2x3::zeros(),
        varying_normals: Matrix3::zeros(),
        varying_view_tri: Matrix3::zeros(),
        varying_shadow_tri: Matrix3::zeros(),
    };

    // Use z_buffer mutably while keeping shadow_buffer immutable
    draw_faces(&model, &mut colour_buffer, &mut z_buffer, &mut shader);

    // Flip the image vertically
    image::imageops::flip_vertical_in_place(&mut colour_buffer);

    colour_buffer
}