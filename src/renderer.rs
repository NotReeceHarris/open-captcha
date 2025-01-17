#![allow(dead_code)]
pub mod gl;
pub mod shaders;

extern crate image;
extern crate nalgebra;
extern crate obj;

use self::shaders::{Shader, RenderingShader, ShadowShader};
use self::gl::{get_model_view_matrix, get_projection_matrix, get_viewport_matrix};

use std::sync::Arc;
use obj::{Obj, TexturedVertex};
use image::{ImageBuffer, RgbaImage, Rgba};
use nalgebra::{clamp, Point2, Point4, Vector2, Vector3, Matrix2x3, Matrix3, Point3};
use std::time::Instant;

const WIDTH: u32 = 250;
const HEIGHT: u32 = 250;

pub struct Camera {
    pub position: Point3<f32>,
    pub focal_length: f32,
    pub view_point: Point3<f32>,
}

/// Implementation of barycentric algorithm for triangle filling. Works as the rasterizer.
fn draw_face_barycentric(
    screen_coords: [Point4<f32>; 3],
    shaders: &dyn Shader,
    color_buffer: &mut RgbaImage,
    z_buffer: &mut [Vec<f32>],
) {
    // Pre-calculate 1/w for perspective correction
    let inv_w = [1.0/screen_coords[0].w, 1.0/screen_coords[1].w, 1.0/screen_coords[2].w];
    
    // Screen coordinates post-perspective division
    let screen_coords2: [Point4<f32>; 3] = [
        screen_coords[0] * inv_w[0],
        screen_coords[1] * inv_w[1],
        screen_coords[2] * inv_w[2],
    ];

    // Clamp coordinates in one pass
    let width = color_buffer.width() as f32 - 1.0;
    let height = color_buffer.height() as f32 - 1.0;
    let screen_coords2 = screen_coords2.map(|mut coord| {
        coord.x = coord.x.clamp(0.0, width);
        coord.y = coord.y.clamp(0.0, height);
        coord
    });
    let [v0_s, v1_s, v2_s] = screen_coords2;

    // Calculate bounding box with integer operations
    let min_x = screen_coords2.iter().map(|v| v.x as i32).min().unwrap();
    let max_x = screen_coords2.iter().map(|v| v.x as i32).max().unwrap();
    let min_y = screen_coords2.iter().map(|v| v.y as i32).min().unwrap();
    let max_y = screen_coords2.iter().map(|v| v.y as i32).max().unwrap();

    // Pre-calculate edge vectors
    let vec1: Vector2<f32> = (v1_s - v0_s).xy();
    let vec2: Vector2<f32> = (v2_s - v0_s).xy();
    let vec1_x_vec2 = vec1.perp(&vec2);
    
    // Early exit if triangle is degenerate
    if vec1_x_vec2.abs() < 1e-6 {
        return;
    }

    let inv_vec1_x_vec2 = 1.0 / vec1_x_vec2;

    // Calculate scanline bounds for better cache coherency
    for y in min_y..=max_y {
        let y_f32 = y as f32;
        
        for x in min_x..=max_x {
            let pv0 = Point2::<f32>::new(x as f32, y_f32) - v0_s.xy();
            
            // Optimized barycentric coordinate calculation
            let vec1_x_pv0 = vec1.perp(&pv0);
            let pv0_x_vec2 = pv0.perp(&vec2);
            
            let s = vec1_x_pv0 * inv_vec1_x_vec2;
            let t = pv0_x_vec2 * inv_vec1_x_vec2;
            
            // Early reject if outside triangle
            if s < 0.0 || t < 0.0 || (s + t) > 1.0 {
                continue;
            }

            let t_s_1 = 1.0 - (t + s);
            
            // Optimized perspective correction
            let w = t_s_1 * inv_w[0] + t * inv_w[1] + s * inv_w[2];
            let inv_total_w = 1.0 / w;

            // Calculate perspective-corrected barycentric coordinates
            let bar_coords = Vector3::new(
                t_s_1 * inv_w[0] * inv_total_w,
                t * inv_w[1] * inv_total_w,
                s * inv_w[2] * inv_total_w
            );

            let z_value = t_s_1 * v0_s.z + t * v1_s.z + s * v2_s.z;
            let buf_idx = x as usize;
            let buf_idy = y as usize;
            
            if z_buffer[buf_idx][buf_idy] <= z_value {
                z_buffer[buf_idx][buf_idy] = z_value;
                if let Some(frag) = shaders.fragment_shader(bar_coords) {
                    color_buffer.put_pixel(x as u32, y as u32, frag);
                }
            }
        }
    }
}

fn get_face_world_coords(model: &Obj<TexturedVertex>, face: &[u16]) -> [Point4<f32>; 3] {
    face[..3].iter().map(|&idx| {
        let [x, y, z] = model.vertices[idx as usize].position;
        Point4::<f32>::new(x, y, z, 1.0)
    }).collect::<Vec<_>>().try_into().unwrap()
}

fn draw_faces(
    model: &Obj<TexturedVertex>,
    color_buffer: &mut RgbaImage,
    z_buffer: &mut [Vec<f32>],
    shaders: &mut dyn Shader,
) {
    model.indices.chunks(3).for_each(|face| {
        let mut verts = get_face_world_coords(model, face);
        for (i, vert) in verts.iter_mut().enumerate() {
            shaders.vertex_shader(face[i], i, vert);
        }
        draw_face_barycentric(verts, shaders, color_buffer, z_buffer);
    });
}

pub fn render(
    model: Arc<Obj<TexturedVertex>>,
    texture: ImageBuffer<Rgba<u8>, Vec<u8>>,
    normal_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
    specular_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> RgbaImage {
    let mut color_buffer = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([0, 0, 0, 255]));
    let mut _buffer = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([0, 0, 0, 255]));

    let perf_total = Instant::now();
    let perf_load = Instant::now();

    // Frame properties
    let (width, height) = (color_buffer.width() as f32, color_buffer.height() as f32);
    let depth = 1024.;

    // Model configuration
    let model_pos = Point3::new(0., 0., 0.);
    let model_scale = Vector3::new(1., 1., 1.);

    // Camera configuration
    let camera = Camera {
        position: Point3::new(0., 0., 2.2),
        focal_length: 3.,
        view_point: model_pos,
    };

    // Light configuration
    let ambient_light = 5.;
    let dir_light = Vector3::new(-1., 0., 1.5);

    // Z buffer
    let mut z_buffer = vec![vec![f32::NEG_INFINITY; height as usize]; width as usize];

    // Shadow buffer
    let mut shadow_buffer = vec![vec![f32::NEG_INFINITY; height as usize]; width as usize];

    // Transformation matrices
    let model_view = get_model_view_matrix(
        camera.position,
        camera.view_point,
        model_pos,
        model_scale,
        Vector3::new(0., 1., 0.),
    );
    let projection = get_projection_matrix(camera.focal_length);
    let model_view_it = model_view.try_inverse().unwrap().transpose();
    let viewport = get_viewport_matrix(height, width, depth);

    let shadow_mat = get_model_view_matrix(
        Point3::<f32>::origin() + dir_light,
        model_pos,
        model_pos,
        model_scale,
        Vector3::new(0., 1., 0.),
    );

    println!("Model loaded in {} ms", perf_load.elapsed().as_millis());
    let perf_shadow = Instant::now();

    let mut shadow_shader = ShadowShader {
        model: &model,
        uniform_shadow_mv_mat: shadow_mat,
        uniform_viewport: viewport,

        varying_view_tri: Matrix3::<f32>::zeros(),
    };
    
    // Compute shadows
    draw_faces(&model, &mut _buffer, &mut shadow_buffer, &mut shadow_shader);

    println!("Shadows computed in {} ms", perf_shadow.elapsed().as_millis());
    let perf_shaders = Instant::now();

    let mut rendering_shader = RenderingShader {
        model: &model,
        shadow_buffer: &shadow_buffer,
        uniform_model_view: model_view,
        uniform_model_view_it: model_view_it,
        uniform_shadow_mv_mat: shadow_mat,
        uniform_projection: projection,
        uniform_viewport: viewport,
        uniform_ambient_light: ambient_light,
        uniform_dir_light: (model_view * dir_light.insert_row(3, 0.)).xyz().normalize(),
        uniform_texture: texture,
        uniform_normal_map: normal_map,
        uniform_specular_map: specular_map,
        varying_uv: Matrix2x3::<f32>::zeros(),
        varying_normals: Matrix3::<f32>::zeros(),
        varying_view_tri: Matrix3::<f32>::zeros(),
        varying_shadow_tri: Matrix3::<f32>::zeros(),
    };

    println!("Shaders computed in {} ms", perf_shaders.elapsed().as_millis());
    let perf_render = Instant::now();

    // Render model
    draw_faces(
        &model,
        &mut color_buffer,
        &mut z_buffer,
        &mut rendering_shader,
    );

    println!("Model rendered in {} ms", perf_render.elapsed().as_millis());
    println!("Frame rendered in {} ms\n----", perf_total.elapsed().as_millis());

    color_buffer
}