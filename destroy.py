
# frequency disruption
# shrink the image to 99% of its size and stretch it back using interpolation. breaking alignment by slightly distorting the grid
# save image as jpg, quality setting of 75. because JPEG compression discards high-frequecy data to strip out watermarks
from PIL import Image, ImageFilter
import random

def attack_synthid(image_path, output_path):
    img = Image.open(image_path)
    
    # 1. Micro-rescaling shift (forces interpolation grid distortion)
    w, h = img.size
    img_resized = img.resize((int(w * 0.99), int(h * 0.99)), Image.Resampling.BICUBIC)
    img_back = img_resized.resize((w, h), Image.Resampling.BICUBIC)
    
    # 2. Aggressive lossy recompression (disrupts high-frequency carrier bins)
    img_back.save(output_path, "JPEG", quality=75)
    print(f"Processed and saved to {output_path}")

if __name__ == "__main__":
    attack_synthid("apple.png", "bypassed.png")