import numpy as np
from PIL import Image

def evaluate_spectral_capping(input_path, threshold=300, dc_radius=15):
    print(f"\n--- Evaluating Spectral Capping ---")
    print(f"Image: {input_path}")
    print(f"Arbitrary Amplitude Cutoff: {threshold} (Not a detected-peak threshold)")
    
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    h, w, _ = arr.shape
    
    reconstructed = np.zeros_like(arr)
    
    print(f"Total FFT bins per channel: {h * w:,}\n")
    
    for i, name in enumerate(["Red", "Green", "Blue"]):
        channel = arr[:, :, i]
        
        # Raw FFT (No window, no zero-mean)
        fft_raw = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft_raw)
        
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # Protect DC
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dc_mask = ((x - cx)**2 + (y - cy)**2) <= dc_radius**2
        
        # Identify all bins exceeding the arbitrary threshold
        capped_mask = (magnitude > threshold) & (~dc_mask)
        affected_count = np.sum(capped_mask)
        affected_fraction = (affected_count / magnitude.size) * 100
        
        # Apply the cap
        magnitude[capped_mask] = threshold
        fft_capped = magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        inverse = np.fft.ifft2(np.fft.ifftshift(fft_capped))
        
        # DSP Sanity Check: Measure imaginary residue to confirm symmetry
        max_imag = np.max(np.abs(inverse.imag))
        
        print(f"{name:<5} | Bins capped: {affected_count:<7,} | Affected fraction: {affected_fraction:05.2f}% | Max imag residue: {max_imag:.3e}")
        
        reconstructed[:, :, i] = inverse.real
        
    # --- Measure Reconstruction Error ---
    
    diff_float = reconstructed - arr
    mae_float = np.mean(np.abs(diff_float))
    max_abs_float = np.max(np.abs(diff_float))
    mse_float = np.mean(diff_float**2)
    rmse_float = np.sqrt(mse_float)
    
    arr_8bit = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    recon_8bit = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
    
    mse_8bit = np.mean((arr_8bit.astype(np.float64) - recon_8bit.astype(np.float64))**2)
    rmse_8bit = np.sqrt(mse_8bit)
    
    print("\n--- Float-Domain Error (Before Quantization) ---")
    print(f"Float RMSE:               {rmse_float:.6f}")
    print(f"Float MAE:                {mae_float:.6f}")
    print(f"Maximum absolute change:  {max_abs_float:.6f}")
    
    print("\n--- 8-Bit Domain Error (After Quantization) ---")
    print(f"8-bit RMSE:               {rmse_8bit:.6f}")
    
    # Calculate changed pixels and sub-pixels
    pixel_changed_mask = np.any(arr_8bit != recon_8bit, axis=2)
    pixels_changed = np.sum(pixel_changed_mask)
    subpixels_changed = np.sum(arr_8bit != recon_8bit)
    total_subpixels = h * w * 3
    
    print(f"RGB pixels changed:       {pixels_changed:,} / {h*w:,} ({(pixels_changed/(h*w))*100:.2f}%)")
    print(f"Sub-pixels changed:       {subpixels_changed:,} / {total_subpixels:,} ({(subpixels_changed/total_subpixels)*100:.2f}%)")

if __name__ == "__main__":
    evaluate_spectral_capping("ai-generated-images/pure-white-blank-canvas1.png", threshold=300)