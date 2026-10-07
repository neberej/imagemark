import numpy as np
from PIL import Image

THRESHOLDS = [300, 500, 750, 1000, 1500, 2000, 2500, 3000]


def evaluate_threshold(arr, threshold, dc_radius=15):
    h, w, c = arr.shape
    reconstructed = np.zeros_like(arr)

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dc_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= dc_radius ** 2

    per_channel_affected = []
    max_imag_residue = 0.0

    for i in range(c):
        channel = arr[:, :, i]

        fft_raw = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft_raw)

        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        capped_mask = (magnitude > threshold) & (~dc_mask)
        affected_count = np.sum(capped_mask)
        affected_fraction = (affected_count / magnitude.size) * 100.0
        per_channel_affected.append(affected_fraction)

        magnitude[capped_mask] = threshold
        fft_capped = magnitude * np.exp(1j * phase)

        inverse = np.fft.ifft2(np.fft.ifftshift(fft_capped))

        max_imag_residue = max(
            max_imag_residue,
            np.max(np.abs(inverse.imag))
        )

        reconstructed[:, :, i] = inverse.real

    overall_affected_fraction = float(np.mean(per_channel_affected))

    # Float-domain error
    diff_float = reconstructed - arr
    rmse_float = np.sqrt(np.mean(diff_float ** 2))
    mae_float = np.mean(np.abs(diff_float))
    max_abs_float = np.max(np.abs(diff_float))

    # 8-bit / saved-image error
    arr_8bit = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    recon_8bit = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

    diff_8bit = arr_8bit.astype(np.float64) - recon_8bit.astype(np.float64)
    rmse_8bit = np.sqrt(np.mean(diff_8bit ** 2))

    pixel_changed_mask = np.any(arr_8bit != recon_8bit, axis=2)
    pixels_changed_pct = (np.sum(pixel_changed_mask) / (h * w)) * 100.0

    subpixels_changed_pct = (
        np.sum(arr_8bit != recon_8bit) / arr_8bit.size
    ) * 100.0

    return {
        "threshold": threshold,
        "affected_overall_pct": overall_affected_fraction,
        "affected_red_pct": per_channel_affected[0],
        "affected_green_pct": per_channel_affected[1],
        "affected_blue_pct": per_channel_affected[2],
        "rmse_float": rmse_float,
        "mae_float": mae_float,
        "max_abs_float": max_abs_float,
        "rmse_8bit": rmse_8bit,
        "pixels_changed_pct": pixels_changed_pct,
        "subpixels_changed_pct": subpixels_changed_pct,
        "max_imag_residue": max_imag_residue,
    }


def run_sweep(input_path, dc_radius=15):
    print("\n--- Running Threshold Sweep ---")
    print(f"Image: {input_path}")
    print(f"DC protection radius: {dc_radius}\n")

    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)

    print(
        f"{'Threshold':<10} | "
        f"{'FFT Avg':<9} | "
        f"{'R%':<7} | "
        f"{'G%':<7} | "
        f"{'B%':<7} | "
        f"{'Float RMSE':<11} | "
        f"{'8-bit RMSE':<11} | "
        f"{'RGB Px':<9} | "
        f"{'Subpx':<9} | "
        f"{'Max Imag'}"
    )
    print("-" * 120)

    for t in THRESHOLDS:
        result = evaluate_threshold(arr, t, dc_radius=dc_radius)

        print(
            f"{result['threshold']:<10} | "
            f"{result['affected_overall_pct']:>7.2f}% | "
            f"{result['affected_red_pct']:>6.2f}% | "
            f"{result['affected_green_pct']:>6.2f}% | "
            f"{result['affected_blue_pct']:>6.2f}% | "
            f"{result['rmse_float']:>10.4f} | "
            f"{result['rmse_8bit']:>10.4f} | "
            f"{result['pixels_changed_pct']:>7.2f}% | "
            f"{result['subpixels_changed_pct']:>7.2f}% | "
            f"{result['max_imag_residue']:.2e}"
        )


if __name__ == "__main__":
    run_sweep(
        "ai-generated-images/pure-white-blank-canvas1.png",
        dc_radius=15,
    )