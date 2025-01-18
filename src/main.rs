mod rendering;
mod background;
mod filters;

use warp::Filter;
use image::{ImageBuffer, Rgba};
use obj::{load_obj, Obj, TexturedVertex};

use rand::seq::SliceRandom;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() {

    let perf_init = std::time::Instant::now();

    let assets = [
        "assets/barrel",
        "assets/head",
        "assets/devil",
    ].iter().map(|dir| {

        let assets_dir = Path::new(dir)
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

        (model, texture, normal_map, specular_map)

    }).collect::<Vec<_>>();

    println!("Init time: {:?}", perf_init.elapsed());

    let assets_filter = warp::any().map(move || assets.clone());

    let index = warp::path::end()
        .and(assets_filter)
        .and_then(serve_captcha);

    println!("Starting server at 'http://localhost:3030/'");
    warp::serve(index).run(([127, 0, 0, 1], 3030)).await;
}

async fn serve_captcha(
    assets: Vec<(Arc<Obj<TexturedVertex>>, ImageBuffer<Rgba<u8>, Vec<u8>>, ImageBuffer<Rgba<u8>, Vec<u8>>, ImageBuffer<Rgba<u8>, Vec<u8>>)>,
) -> Result<impl warp::Reply, warp::Rejection> {

    // select a random model
    let (model, texture, normal_map, specular_map) = assets.choose(&mut rand::thread_rng()).unwrap();


    // Render
    let perf_render = std::time::Instant::now();
    let color_buffer = rendering::render(model.clone(), texture.clone(), normal_map.clone(), specular_map.clone());
    println!("Render time: {:?} \n----", perf_render.elapsed());

    // Convert the image to PNG bytes
    let mut render_buf = std::io::Cursor::new(Vec::new());
    color_buffer
        .write_to(&mut render_buf, image::ImageFormat::Png)
        .unwrap();

    // Generate a random background
    let bg = background::generate(200, 200);

    // layer the background and the rendered image
    let mut bg_img = image::load_from_memory(&bg).unwrap().to_rgba8();
    image::imageops::overlay(&mut bg_img, &color_buffer, 0, 0);

    // Convert the image to PNG bytes
    let mut frame_image = std::io::Cursor::new(Vec::new());
    bg_img
        .write_to(&mut frame_image, image::ImageFormat::Png)
        .unwrap();

    let epsilon = 0.05; // Small perturbation
    let filtered = filters::process_adversarial_image(frame_image.clone(), epsilon);

    // Return the PNG bytes as a response
    Ok(
        warp::reply::with_header(
            filtered.into_inner(),
            "content-type",
            "image/png",
        ),
    )
}
