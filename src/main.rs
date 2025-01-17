mod renderer;

use image::{ImageBuffer, Rgba};
use obj::{load_obj, Obj, TexturedVertex};
use renderer::render;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use warp::Filter;

#[tokio::main]
async fn main() {
    let assets_dir = Path::new("assets/barrel")
        .canonicalize()
        .unwrap_or_else(|_| panic!("Wrong path for assets directory!"));

    // Load model
    let obj_path = assets_dir.join("obj.obj");
    let input = BufReader::new(File::open(obj_path).unwrap());
    let model: Obj<TexturedVertex> = load_obj(input).unwrap();
    let model = Arc::new(model);

    // Load texture
    let texture_path = assets_dir.join("diffuse.tga");
    let mut texture = image::open(texture_path)
        .expect("Opening image failed")
        .into_rgba8();
    image::imageops::flip_vertical_in_place(&mut texture);

    // Load normal map
    let normal_map_path = assets_dir.join("tangent.tga");
    let mut normal_map = image::open(normal_map_path)
        .expect("Opening image failed")
        .into_rgba8();
    image::imageops::flip_vertical_in_place(&mut normal_map);

    // Load specular map
    let specular_map_path = assets_dir.join("spec.tga");
    let mut specular_map = image::open(specular_map_path)
        .expect("Opening image failed")
        .into_rgba8();
    image::imageops::flip_vertical_in_place(&mut specular_map);

    let model_filter = warp::any().map(move || Arc::clone(&model));
    let texture_filter = warp::any().map(move || texture.clone());
    let normal_map_filter = warp::any().map(move || normal_map.clone());
    let specular_map_filter = warp::any().map(move || specular_map.clone());

    let index = warp::path::end()
        .and(model_filter)
        .and(texture_filter)
        .and(normal_map_filter)
        .and(specular_map_filter)
        .and_then(serve_captcha);

    warp::serve(index).run(([127, 0, 0, 1], 3030)).await;
}

async fn serve_captcha(
    model: Arc<Obj<TexturedVertex>>,
    texture: ImageBuffer<Rgba<u8>, Vec<u8>>,
    normal_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
    specular_map: ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    // Render
    let perf_render = std::time::Instant::now();
    let color_buffer = render(model, texture, normal_map, specular_map);
    println!("Render time: {:?}", perf_render.elapsed());

    // Convert the image to PNG bytes
    let mut buf = std::io::Cursor::new(Vec::new());
    color_buffer
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();

    // Return the PNG bytes as a response
    Ok(warp::reply::with_header(
        buf.into_inner(),
        "Content-Type",
        "image/png",
    ))
}
