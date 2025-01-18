use image::{ImageReader, DynamicImage, ImageFormat};
use ndarray::Array3;
use std::io::{Cursor, Seek, SeekFrom};

fn load_image_from_buffer(buffer: Cursor<Vec<u8>>) -> Array3<f32> {
    let mut buffer = buffer;

    // Reset the cursor to the start of the buffer (if it is already at the end, this ensures we can read it again)
    buffer.seek(SeekFrom::Start(0)).expect("Failed to seek to start of buffer");

    let img = ImageReader::new(buffer)
        .with_guessed_format() // This should guess the format correctly if the data is PNG
        .expect("Failed to guess image format")
        .decode() // Decode the image
        .expect("Failed to decode image")
        .to_rgb8(); // Convert to RGB8 (standard RGB format)

    let (width, height) = img.dimensions();
    let mut array = Array3::<f32>::zeros((height as usize, width as usize, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        array[[y as usize, x as usize, 0]] = r as f32 / 255.0; // Normalize to [0, 1]
        array[[y as usize, x as usize, 1]] = g as f32 / 255.0;
        array[[y as usize, x as usize, 2]] = b as f32 / 255.0;
    }
    array
}

/// Save an ndarray as a PNG buffer and return it as Cursor<Vec<u8>>
fn save_ndarray_to_buffer(array: &Array3<f32>) -> Cursor<Vec<u8>> {
    let (height, width, _) = array.dim();
    let mut img = image::ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (array[[y, x, 0]].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (array[[y, x, 1]].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (array[[y, x, 2]].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    let mut output_buffer = Vec::new();
    let mut cursor = Cursor::new(&mut output_buffer);

    // Use the write_to function with the cursor
    DynamicImage::ImageRgb8(img)
        .write_to(&mut cursor, ImageFormat::Png)
        .expect("Failed to write image to buffer");

    // Return the Cursor<Vec<u8>>
    Cursor::new(output_buffer)
}

/// Generate adversarial perturbations
fn generate_adversarial_pattern(
    input: &Array3<f32>,
    gradients: &Array3<f32>,
    epsilon: f32,
) -> Array3<f32> {
    // Apply FGSM: perturbation = epsilon * sign(gradient)
    input + &gradients.mapv(|g| g.signum() * epsilon)
}

/// Mock gradient computation (Replace with your ML model's gradient calculation)
fn compute_mock_gradients(image: &Array3<f32>) -> Array3<f32> {
    // Example: Gradient that highlights edges
    let (height, width, _) = image.dim();
    let mut gradients = Array3::<f32>::zeros((height, width, 3));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            gradients[[y, x, 0]] = image[[y + 1, x, 0]] - image[[y - 1, x, 0]];
            gradients[[y, x, 1]] = image[[y, x + 1, 1]] - image[[y, x - 1, 1]];
            gradients[[y, x, 2]] = image[[y + 1, x + 1, 2]] - image[[y - 1, x - 1, 2]];
        }
    }
    gradients
}

pub fn process_adversarial_image(input_buffer: Cursor<Vec<u8>>, epsilon: f32) -> Cursor<Vec<u8>> {
    // Load the image from the input buffer
    let image = load_image_from_buffer(input_buffer);

    // Mock gradient computation (Replace with actual gradients from a model)
    let gradients = compute_mock_gradients(&image);

    // Generate adversarial pattern
    let adversarial_image = generate_adversarial_pattern(&image, &gradients, epsilon);

    // Save the perturbed image to an output buffer and return as Cursor<Vec<u8>>
    save_ndarray_to_buffer(&adversarial_image)
}
