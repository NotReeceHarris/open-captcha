#![allow(non_snake_case)]

#[macro_use]
extern crate derive_builder;

pub mod math;
pub mod parsing;
pub mod plane_buffer;
pub mod ui;
pub mod visual;
pub mod wavefront;

use ui::render_window::render_window::open_render_window;
use wavefront::{wavefront_obj::WavefrontObj, wavefront_obj_source::WaveFrontObjSource};

const BUFFER_WIDTH: usize = 1000;
const BUFFER_HEIGHT: usize = 1000;
const Z_BUFFER_SIZE: f32 = 255.0;

const MODEL: WaveFrontObjSource = WaveFrontObjSource::new(
    "./resources/taru.obj",
    "./resources/taru_Diffuse.tga",
    None,
    None,
    None,
);

fn main() -> Result<(), String> {
    let wavefront = WavefrontObj::from_sources_struct(&MODEL)?;

    open_render_window(
        BUFFER_WIDTH,
        BUFFER_HEIGHT,
        Z_BUFFER_SIZE,
        vec![wavefront.into()],
    );

    Ok(())
}
