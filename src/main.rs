mod rendering;

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

    #[cfg(debug_assertions)]
    let perf_select = std::time::Instant::now();
    
    // select a random model
    let (model, texture, normal_map, specular_map) = assets.choose(&mut rand::thread_rng()).unwrap();
    println!("Select time: {:?}", perf_select.elapsed());


    // Render
    let perf_render = std::time::Instant::now();
    let color_buffer = rendering::render(model.clone(), texture.clone(), normal_map.clone(), specular_map.clone());

    #[cfg(debug_assertions)]
    println!("Render time: {:?} \n----", perf_render.elapsed());

    // Convert the image to PNG bytes
    let mut buf = std::io::Cursor::new(Vec::new());
    color_buffer
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();

    // Return the PNG bytes as a response
    Ok(
        warp::reply::with_header(
            buf.into_inner(),
            "content-type",
            "image/png",
        ),
    )
}
