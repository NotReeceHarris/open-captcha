use warp::Filter;
use rand::Rng;
use image::{ImageBuffer, Rgba};
use std::io::Cursor; // Import Cursor for the fix

#[tokio::main]
async fn main() {
    // Define the index route
    let index = warp::path::end().and_then(serve_captcha);

    // Start the Warp server
    warp::serve(index).run(([127, 0, 0, 1], 3030)).await;
}

async fn serve_captcha() -> Result<impl warp::Reply, warp::Rejection> {
    // Generate a random colour for the background
    let mut rng = rand::thread_rng();
    let bg_r: u8 = rng.gen();
    let bg_g: u8 = rng.gen();
    let bg_b: u8 = rng.gen(); 

    // Generate the image (placeholder for FBX rendering)
    let img = generate_placeholder_image(bg_r, bg_g, bg_b);

    // Convert the image to PNG bytes
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    let png_data = buf.into_inner();

    // Return the image as a response
    Ok(warp::http::Response::builder()
        .header("Content-Type", "image/png")
        .body(png_data)
        .unwrap())
}

fn generate_placeholder_image(r: u8, g: u8, b: u8) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let width: u32 = 256;
    let height = 256;

    let mut img = ImageBuffer::new(width, height);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let intensity = (x ^ y) as u8; // Simple pattern
        *pixel = Rgba([r, g, b, intensity]);
    }

    img
}
