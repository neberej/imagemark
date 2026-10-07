# redraw the image's pixel structure using Image-to-Image (Stable Diffusion)
# destroying watermarks while keeping the original content intact
# take original image, convert to numbers (latents), add some noise, then denoise based on a prompt
# prompt is a compass, as you clean up make sure final result represent an apple
# only strength allow make changes, 0.25 is low, microspocic pixel patterns
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def strip_synthid_via_diffusion(input_path, output_path):
    # Load model (using a standard efficient pipeline like Stable Diffusion v1.5 or similar)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Use MPS (Metal Performance Shaders) for Apple Silicon Macs
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    )
    pipe = pipe.to(device)

    # Load watermarked image and scale to standard 512x512 multiple
    init_image = Image.open(input_path)
    if init_image.mode in ("RGBA", "LA") or (init_image.mode == "P" and "transparency" in init_image.info):
        background = Image.new("RGB", init_image.size, (255, 255, 255))
        background.paste(init_image, mask=init_image.split3[-1] if init_image.mode == "RGBA" else None)
        init_image = background
    else:
        init_image = init_image.convert("RGB")
    init_image = init_image.resize((512, 512))

    # Low denoise strength (e.g., 0.2 to 0.3) scrambles the watermark 
    # while keeping the original image layout and content intact.
    prompt = "A high-quality, detailed, photorealistic macro photograph of a single glossy red apple with a brown stem, natural yellow streaks, isolated on a plain white background with a soft shadow"
    image = pipe(
        prompt=prompt, 
        image=init_image, 
        strength=0.25, 
        guidance_scale=7.5
    ).images[0]

    # Save the processed image
    image.save(output_path)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    strip_synthid_via_diffusion("apple.png", "output_clean.jpg")