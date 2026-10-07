import numpy as np
from PIL import Image

def export_sparse_ablated_image(input_path, output_path="sparse_ablated.png", threshold=2500, dc_radius=15):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    h, w, c = arr.shape
    reconstructed = np.zeros_like(arr)
    
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dc_mask = ((x - cx)**2 + (y - cy)**2) <= dc_radius**2
    
    total_capped = 0
    
    for i in range(c):
        channel = arr[:, :, i]
        fft_raw = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft_raw)
        
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        capped_mask = (magnitude > threshold) & (~dc_mask)
        capped_count = np.sum(capped_mask)
        total_capped += capped_count
        
        print(f"Channel {i}: Capped {capped_count:,} outlier peaks.")
        
        magnitude[capped_mask] = threshold
        fft_capped = magnitude * np.exp(1j * phase)
        
        inverse = np.fft.ifft2(np.fft.ifftshift(fft_capped))
        reconstructed[:, :, i] = inverse.real
        
    print(f"\nTotal outlier bins flattened across all channels: {total_capped:,}")
    
    # Save the 8-bit quantized image
    final_8bit = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
    Image.fromarray(final_8bit).save(output_path)
    print(f"Saved sparse ablated image to: {output_path}")

if __name__ == "__main__":
    export_sparse_ablated_image("ai-generated-images/pure-white-blank-canvas1.png", threshold=2500)