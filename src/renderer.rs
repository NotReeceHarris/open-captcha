#![allow(dead_code)]

pub mod gl;
pub mod shaders;

extern crate image;
extern crate nalgebra;
extern crate obj;

use self::shaders::{Shader, RenderingShader};
use self::gl::{get_model_view_matrix, get_projection_matrix, get_viewport_matrix};

use std::sync::Arc;
use obj::{Obj, TexturedVertex};
use image::{ImageBuffer, RgbaImage, Rgba};
use nalgebra::{Point2, Point3, Point4, Vector3, Matrix2x3, Matrix3};

const WIDTH: u32 = 200;
const HEIGHT: u32 = 200;

/// Represents the camera properties used for rendering
pub struct Camera {
    pub position: Point3<f32>,
    pub focal_length: f32,
    pub view_point: Point3<f32>,
}

/// Rasterises a triangle using barycentric coordinates for triangle filling
fn draw_face_barycentric(
    screen_coords: [Point4<f32>; 3],
    shader: &dyn Shader,
    colour_buffer: &mut RgbaImage,
    z_buffer: &mut [Vec<f32>],
) {
    // Pre-calculate 1/w for perspective correction
    let inv_w = screen_coords.map(|coord| 1.0 / coord.w);

    // Apply perspective correction
    let screen_coords = screen_coords
        .iter()
        .zip(inv_w.iter())
        .map(|(coord, &inv_w)| coord * inv_w)
        .collect::<Vec<_>>();

    // Clamp coordinates within the screen boundaries
    let (width, height) = (colour_buffer.width() as f32 - 1.0, colour_buffer.height() as f32 - 1.0);
    let screen_coords: Vec<Point4<f32>> = screen_coords
        .into_iter()
        .map(|mut coord| {
            coord.x = coord.x.clamp(0.0, width);
            coord.y = coord.y.clamp(0.0, height);
            coord
        })
        .collect();

    let [v0_s, v1_s, v2_s] = [screen_coords[0], screen_coords[1], screen_coords[2]];

    // Calculate bounding box
    let min_x = screen_coords.iter().map(|v| v.x as i32).min().unwrap();
    let max_x = screen_coords.iter().map(|v| v.x as i32).max().unwrap();
    let min_y = screen_coords.iter().map(|v| v.y as i32).min().unwrap();
    let max_y = screen_coords.iter().map(|v| v.y as i32).max().unwrap();

    // Edge vectors and their cross product
    let vec1 = (v1_s - v0_s).xy();
    let vec2 = (v2_s - v0_s).xy();
    let vec1_cross_vec2 = vec1.perp(&vec2);

    // Early exit if the triangle is degenerate
    if vec1_cross_vec2.abs() < 1e-6 {
        return;
    }

    let inv_vec1_cross_vec2 = 1.0 / vec1_cross_vec2;

    // Rasterisation loop
    for y in min_y..=max_y {
        let y_f32 = y as f32;
        for x in min_x..=max_x {
            let pv0 = Point2::new(x as f32, y_f32) - v0_s.xy();
            let vec1_cross_pv0 = vec1.perp(&pv0);
            let pv0_cross_vec2 = pv0.perp(&vec2);

            let s = vec1_cross_pv0 * inv_vec1_cross_vec2;
            let t = pv0_cross_vec2 * inv_vec1_cross_vec2;

            // Check if the point is inside the triangle
            if s < 0.0 || t < 0.0 || (s + t) > 1.0 {
                continue;
            }

            let one_minus_st = 1.0 - (s + t);
            let w = one_minus_st * inv_w[0] + t * inv_w[1] + s * inv_w[2];
            let inv_total_w = 1.0 / w;

            // Perspective-corrected barycentric coordinates
            let bar_coords = Vector3::new(
                one_minus_st * inv_w[0] * inv_total_w,
                t * inv_w[1] * inv_total_w,
                s * inv_w[2] * inv_total_w,
            );

            let z_value = one_minus_st * v0_s.z + t * v1_s.z + s * v2_s.z;
            let buf_x = x as usize;
            let buf_y = y as usize;

            if z_buffer[buf_x][buf_y] <= z_value {
                z_buffer[buf_x][buf_y] = z_value;
                if let Some(frag) = shader.fragment_shader(bar_coords) {
                    colour_buffer.put_pixel(x as u32, y as u32, frag);
                }
            }
        }
    }
}

/// Converts face indices to world coordinates
fn get_face_world_coords(
    model: &Obj<TexturedVertex>,
    face: &[u16],
) -> [Point4<f32>; 3] {
    face.iter()
        .map(|&idx| {
            let [x, y, z] = model.vertices[idx as usize].position;
            Point4::new(x, y, z, 1.0)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Draws all faces of the model
fn draw_faces(
    model: &Obj<TexturedVertex>,
    colour_buffer: &mut RgbaImage,
    z_buffer: &mut [Vec<f32>],
    shader: &mut dyn Shader,
) {
    for face in model.indices.chunks(3) {
        let mut verts = get_face_world_coords(model, face);
        for (i, vert) in verts.iter_mut().enumerate() {
            shader.vertex_shader(face[i], i, vert);
        }
        draw_face_barycentric(verts, shader, colour_buffer, z_buffer);
    }
}

/// Renders the 3D model with shadows and lighting
pub fn render(
    model: Arc<Obj<TexturedVertex>>,
    texture: ImageBuffer<Rgba<u8>, Vec<u8>>,
    normal_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
    specular_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> RgbaImage {
    let mut colour_buffer = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([0, 0, 0, 255]));
    let mut z_buffer = vec![vec![f32::NEG_INFINITY; HEIGHT as usize]; WIDTH as usize];

    // Create a shadow buffer for the shader
    let shadow_buffer = z_buffer.clone();

    // Configure camera and transformations
    let camera = Camera {
        position: Point3::new(0., 0., 2.2),
        focal_length: 3.0,
        view_point: Point3::origin(),
    };

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

    colour_buffer
}