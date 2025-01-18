#![allow(dead_code)]

extern crate image;
extern crate nalgebra;
extern crate obj;

use crate::rendering::shaders::Shader;

use obj::{Obj, TexturedVertex};
use image::RgbaImage;
use nalgebra::{Point2, Point4, Vector3};

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
pub fn draw_faces(
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