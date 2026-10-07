import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def analyze_synthid_white_background(ai_image_path):
    # 1. Load the AI-generated white image
    # Note: If testing a large image, crop it first, do not resize!
    img = Image.open(ai_image_path).convert('RGB')
    ai_array = np.array(img, dtype=np.int16) # int16 prevents underflow during subtraction
    
    # 2. Create the pure white control matrix (255, 255, 255)
    control_array = np.full_like(ai_array, 255, dtype=np.int16)
    
    # 3. Calculate the Pixel Delta
    # Subtracting the AI image from pure white. Any pixel > 0 is the embedded signal.
    delta_array = np.abs(control_array - ai_array)
    
    # 4. Amplify the Residual so it becomes visible to the human eye
    # Multiplying the microscopic variance (e.g., 1 or 2) by 50 to make it obvious
    amplification_factor = 50
    amplified_residual = np.clip(delta_array * amplification_factor, 0, 255).astype(np.uint8)
    
    # 5. Frequency Analysis (FFT) on the Green Channel
    # Research shows the green channel holds the strongest SynthID carrier signal
    green_delta = delta_array[:, :, 1]
    
    # Compute 2D Fast Fourier Transform and shift the zero-frequency to the center
    fft_result = np.fft.fft2(green_delta)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate magnitude spectrum (log scale for visibility)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1e-8)
    
    # 6. Visualize the 3-Step Proof
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].set_title("1. Original AI Image (Appears White)")
    axes[0].imshow(img)
    axes[0].axis('off')
    
    axes[1].set_title(f"2. Amplified Residual (x{amplification_factor})")
    axes[1].imshow(amplified_residual)
    axes[1].axis('off')
    
    axes[2].set_title("3. FFT Frequency Spectrum")
    axes[2].imshow(magnitude_spectrum, cmap='magma')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("synthid_proof.png", dpi=150, bbox_inches='tight')
    print("Success: Analysis saved to 'synthid_proof.png'")

if __name__ == "__main__":
    # Replace with the path to your AI-generated white background
    analyze_synthid_white_background("Pure White Canvas.png")