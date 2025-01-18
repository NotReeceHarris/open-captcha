use image::{Pixel, Rgba, RgbaImage};
use nalgebra::{Matrix2x3, Matrix3, Matrix4, Point2, Point4, Vector2, Vector3, Vector4};
use obj::{Obj, TexturedVertex};

pub trait Shader {
    fn vertex_shader(&mut self, face: u16, nthvert: usize, gl_position: &mut Point4<f32>);
    fn fragment_shader(&self, bar_coords: Vector3<f32>) -> Option<Rgba<u8>>;
}

#[inline(always)]
fn sample_2d(texture: &RgbaImage, uv: Point2<f32>) -> Rgba<u8> {
    let width = texture.width() as f32;
    let height = texture.height() as f32;

    // Precompute coordinates
    let x = (uv.x * width - 1.0).clamp(0.0, width - 1.0) as u32;
    let y = (uv.y * height - 1.0).clamp(0.0, height - 1.0) as u32;

    // Return the pixel without dereferencing
    texture.get_pixel(x, y).clone()
}

pub struct RenderingShader<'a> {
    pub model: &'a Obj<TexturedVertex>,
    pub shadow_buffer: &'a [Vec<f32>],
    pub uniform_model_view: Matrix4<f32>,
    pub uniform_model_view_it: Matrix4<f32>,
    pub uniform_shadow_mv_mat: Matrix4<f32>,
    pub uniform_projection: Matrix4<f32>,
    pub uniform_viewport: Matrix4<f32>,
    pub uniform_ambient_light: f32,
    pub uniform_dir_light: Vector3<f32>,
    pub uniform_texture: RgbaImage,
    pub uniform_specular_map: RgbaImage,
    pub uniform_normal_map: RgbaImage,

    pub varying_uv: Matrix2x3<f32>,
    pub varying_normals: Matrix3<f32>,
    pub varying_view_tri: Matrix3<f32>,
    pub varying_shadow_tri: Matrix3<f32>,
}

impl Shader for RenderingShader<'_> {
    #[inline(always)]
    fn vertex_shader(&mut self, face_idx: u16, nthvert: usize, gl_position: &mut Point4<f32>) {
        let vertex = &self.model.vertices[face_idx as usize];

        // Cache UV coordinates
        let [u, v, _] = vertex.texture;
        self.varying_uv.set_column(nthvert, &Vector2::new(u, v));

        // Transform and normalise normal vector
        let [i, j, k] = vertex.normal;
        let normal = (self.uniform_model_view_it * Vector4::new(i, j, k, 0.))
            .xyz()
            .normalize();
        self.varying_normals.set_column(nthvert, &normal);

        // Shadow buffer transformation
        let mut shadow_pos = Point4::from(self.uniform_shadow_mv_mat * gl_position.coords);
        shadow_pos.x = shadow_pos.x.clamp(-1.0, 1.0);
        shadow_pos.y = shadow_pos.y.clamp(-1.0, 1.0);
        shadow_pos = Point4::from(self.uniform_viewport * shadow_pos.coords);
        self.varying_shadow_tri
            .set_column(nthvert, &shadow_pos.xyz().coords);

        // Process vertex in view coordinates
        let mv_coords = self.uniform_model_view * gl_position.coords;
        self.varying_view_tri
            .set_column(nthvert, &gl_position.xyz().coords);

        // Final transformation
        *gl_position = Point4::from(self.uniform_viewport * self.uniform_projection * mv_coords);
    }

    #[inline(always)]
    fn fragment_shader(&self, bar_coords: Vector3<f32>) -> Option<Rgba<u8>> {
        // Texture coordinates
        let uv = Point2::<f32>::from(self.varying_uv * bar_coords);

        // Shadow buffer calculation
        let shad_buf_p = (self.varying_shadow_tri * bar_coords).insert_row(3, 1.);
        const SHADOW_TOLERANCE: f32 = 10.0;
        let shadow = if self.shadow_buffer[shad_buf_p.x as usize][shad_buf_p.y as usize]
            < shad_buf_p.z + SHADOW_TOLERANCE
        {
            1.0
        } else {
            0.1
        };

        // Normal interpolation and tangent space normal mapping
        let bnormal = self.varying_normals * bar_coords;
        let a = Matrix3::from_rows(&[
            (self.varying_view_tri.column(1) - self.varying_view_tri.column(0)).transpose(),
            (self.varying_view_tri.column(2) - self.varying_view_tri.column(0)).transpose(),
            bnormal.transpose(),
        ])
        .try_inverse()
        .unwrap();

        let i = a * Vector3::new(
            self.varying_uv.column(1)[0] - self.varying_uv.column(0)[0],
            self.varying_uv.column(2)[0] - self.varying_uv.column(0)[0],
            0.0,
        );
        let j = a * Vector3::new(
            self.varying_uv.column(1)[1] - self.varying_uv.column(0)[1],
            self.varying_uv.column(2)[1] - self.varying_uv.column(0)[1],
            0.0,
        );

        let b_mat = Matrix3::from_columns(&[i.normalize(), j.normalize(), bnormal]);

        let Rgba([x, y, z, _]) = sample_2d(&self.uniform_normal_map, uv);
        let darboux_mapping = Vector3::new(x, y, z).map(|w| ((w as f32 / 255.0) * 2.0) - 1.0);
        let normal = b_mat * darboux_mapping;

        // Lighting
        let reflected = (normal * (normal.dot(&self.uniform_dir_light) * 2.0)
            - self.uniform_dir_light)
            .normalize();
        let specular = f32::powi(
            f32::max(0.0, reflected.z),
            (1 + sample_2d(&self.uniform_specular_map, uv)[0]) as i32,
        );
        let diffuse = f32::max(0.0, self.uniform_dir_light.dot(&normal));

        // Fragment colour
        let mut gl_frag_color = sample_2d(&self.uniform_texture, uv);
        gl_frag_color.apply_without_alpha(|ch| {
            (self.uniform_ambient_light + (ch as f32) * shadow * (diffuse + 0.6 * specular)) as u8
        });

        Some(gl_frag_color)
    }
}

/* pub struct ShadowShader<'a> {
    pub model: &'a Obj<TexturedVertex>,
    pub uniform_shadow_mv_mat: Matrix4<f32>,
    pub uniform_viewport: Matrix4<f32>,

    pub varying_view_tri: Matrix3<f32>,
}

impl Shader for ShadowShader<'_> {
    fn vertex_shader(&mut self, _face_idx: u16, nthvert: usize, gl_position: &mut Point4<f32>) {
        *gl_position = Point4::from(self.uniform_shadow_mv_mat * gl_position.coords);
        gl_position.x = clamp(gl_position.x, -1.0, 1.0);
        gl_position.y = clamp(gl_position.y, -1.0, 1.0);
        *gl_position = Point4::from(self.uniform_viewport * gl_position.coords);
        self.varying_view_tri
            .set_column(nthvert, &gl_position.xyz().coords);
    }
    fn fragment_shader(&self, _bar_coords: Vector3<f32>) -> Option<Rgba<u8>> {
        None
    }
}*/
