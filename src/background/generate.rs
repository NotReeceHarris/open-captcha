use image::{ImageBuffer, Rgb};
use rand::Rng;
use std::io::Cursor;

// Generates a random background and outputs it as PNG bytes.
pub fn generate(width: u32, height: u32) -> Vec<u8> {
    // Create an ImageBuffer with RGB pixels
    let mut img = ImageBuffer::new(width, height);

    // Fill the image with the random colour
    for pixel in img.pixels_mut() {

        let mut rng = rand::thread_rng();
        let red: u8 = rng.gen();
        let green: u8 = rng.gen();
        let blue: u8 = rng.gen();

        *pixel = Rgb([red, green, blue]);
    }

    // Save the image to a PNG in-memory buffer using Cursor
    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, image::ImageFormat::Png)
        .expect("Failed to write image to buffer");

    buffer.into_inner() // Extract the Vec<u8> from the Cursor
}