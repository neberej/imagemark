import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return img, arr


def fft_magnitude(channel):
    # Remove DC component so the mean brightness offset does not dominate.
    centered = channel - channel.mean()

    # Reduce edge/boundary artifacts in the FFT.
    h, w = centered.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = centered * window

    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)

    # log1p is numerically stable and easier to visualize.
    return np.log1p(np.abs(fft_shifted))


def print_channel_stats(name, channel):
    print(
        f"{name:>5} | "
        f"mean={channel.mean():8.4f} "
        f"std={channel.std():8.4f} "
        f"min={channel.min():6.1f} "
        f"max={channel.max():6.1f}"
    )


def analyze_and_compare(ai_path, reference_path, output_path="comparison_result.png"):
    ai_img, ai = load_rgb(ai_path)
    ref_img, ref = load_rgb(reference_path)

    if ai.shape != ref.shape:
        raise ValueError(
            f"Image dimensions differ: AI={ai.shape}, reference={ref.shape}"
        )

    print("--- AI Image Pixel Analysis ---")
    print(f"Shape: {ai.shape}")
    print(f"Min pixel value: {ai.min():.0f}")
    print(f"Max pixel value: {ai.max():.0f}")
    print(f"Mean pixel value: {ai.mean():.4f}")

    unique_values = np.unique(ai.astype(np.uint8))
    print(f"Number of unique channel values: {len(unique_values)}")

    if len(unique_values) <= 30:
        print(f"Unique channel values: {unique_values}")
    else:
        print(
            f"Unique channel value range: "
            f"{unique_values[0]} ... {unique_values[-1]}"
        )

    if np.all(ai == 255):
        print("\n[RESULT] All decoded RGB pixels are exactly 255.")
        print("There is no RGB pixel-domain residual to analyze.")
        return

    print("\n[RESULT] The AI image is not mathematically pure white.")

    # Signed difference preserves the direction of the residual.
    signed_delta = ai - ref

    # Absolute difference is useful only for visualization.
    abs_delta = np.abs(signed_delta)

    print("\n--- Residual Statistics ---")
    print(f"Absolute mean error: {abs_delta.mean():.6f}")
    print(f"Maximum absolute error: {abs_delta.max():.1f}")
    print(f"Residual RMS: {np.sqrt(np.mean(signed_delta ** 2)):.6f}")

    print("\n--- Signed Residual by Channel ---")
    print_channel_stats("Red", signed_delta[:, :, 0])
    print_channel_stats("Green", signed_delta[:, :, 1])
    print_channel_stats("Blue", signed_delta[:, :, 2])

    # Amplify only for human visualization.
    amp_factor = 50
    amplified = np.clip(abs_delta * amp_factor, 0, 255).astype(np.uint8)

    red_fft = fft_magnitude(signed_delta[:, :, 0])
    green_fft = fft_magnitude(signed_delta[:, :, 1])
    blue_fft = fft_magnitude(signed_delta[:, :, 2])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].set_title("AI-generated image")
    axes[0, 0].imshow(ai_img)
    axes[0, 0].axis("off")

    axes[0, 1].set_title("Reference image")
    axes[0, 1].imshow(ref_img)
    axes[0, 1].axis("off")

    axes[0, 2].set_title(f"Absolute residual ×{amp_factor}")
    axes[0, 2].imshow(amplified)
    axes[0, 2].axis("off")

    axes[1, 0].set_title("FFT: red residual")
    axes[1, 0].imshow(red_fft, cmap="magma")
    axes[1, 0].axis("off")

    axes[1, 1].set_title("FFT: green residual")
    axes[1, 1].imshow(green_fft, cmap="magma")
    axes[1, 1].axis("off")

    axes[1, 2].set_title("FFT: blue residual")
    axes[1, 2].imshow(blue_fft, cmap="magma")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved analysis to: {output_path}")

if __name__ == "__main__":
    analyze_and_compare(
        "Pure-White-Canvas.png",
        "pure_white_full.png",
        "comparison_result.png",
    )