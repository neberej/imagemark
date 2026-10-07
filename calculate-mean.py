import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Compute a windowed, 0 mean 2D FFT spectrum
def fft_spectrum(channel):
    centered = channel - channel.mean(dtype=np.float64) # zero mean
    h, w = centered.shape
    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    )
    windowed = centered * window
    fft_raw = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_raw)
    magnitude = np.log1p(np.abs(fft_shifted))
    return magnitude

# analyze spatial variation
# load image, calculate mean RGB independently, subtract mean color from every pixel and compute FFT spectra
def analyze_zero_mean_residual(
    image_path,
    output_path="zero_mean_analysis.png",
    amp_factor=50,
):
    img = Image.open(image_path).convert("RGB")
    ai = np.array(img, dtype=np.float64)
    height, width, channels = ai.shape

    print("--- Image ---")
    print(f"Path: {image_path}")
    print(f"Dimensions: {width} x {height}")
    print(f"Shape: {ai.shape}")
    print("--- Raw Pixel Statistics ---")
    print(f"Overall min:  {ai.min():.0f}")
    print(f"Overall max:  {ai.max():.0f}")
    print(f"Overall mean: {ai.mean(dtype=np.float64):.6f}")

    unique_values = np.unique(ai.astype(np.uint8))
    print(f"Unique channel values: {len(unique_values)}")

    if len(unique_values) <= 30:
        print(f"Values: {unique_values}")
    else:
        print(
            f"Value range: "
            f"{unique_values[0]} ... {unique_values[-1]}"
        )

    mean_rgb = ai.mean(
        axis=(0, 1),
        dtype=np.float64,
    )

    print("--- Mean RGB ---")
    print(f"R = {mean_rgb[0]:.9f}")
    print(f"G = {mean_rgb[1]:.9f}")
    print(f"B = {mean_rgb[2]:.9f}")

    channel_min = ai.min(axis=(0, 1))
    channel_max = ai.max(axis=(0, 1))

    print("--- Channel Ranges ---")
    print(
        f"R: {channel_min[0]:.0f} to {channel_max[0]:.0f}"
    )
    print(
        f"G: {channel_min[1]:.0f} to {channel_max[1]:.0f}"
    )
    print(
        f"B: {channel_min[2]:.0f} to {channel_max[2]:.0f}"
    )

    if np.any(mean_rgb < channel_min) or np.any(mean_rgb > channel_max):
        raise RuntimeError(
            "Calculated RGB mean is outside the observed pixel range."
            f"Mean: {mean_rgb}"
            f"Min:  {channel_min}"
            f"Max:  {channel_max}"
        )

    residual = ai - mean_rgb
    residual_channel_means = residual.mean(
        axis=(0, 1),
        dtype=np.float64,
    )

    print("--- Residual ---")
    print(f"Min:  {residual.min():.9f}")
    print(f"Max:  {residual.max():.9f}")
    print(
        f"Overall mean: "
        f"{residual.mean(dtype=np.float64):.12f}"
    )

    print("--- Residual Mean By Channel ---")
    print(f"R = {residual_channel_means[0]:.12f}")
    print(f"G = {residual_channel_means[1]:.12f}")
    print(f"B = {residual_channel_means[2]:.12f}")

    if not np.allclose(
        residual_channel_means,
        0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Residual was not centered correctly."
            f"Residual means: {residual_channel_means}"
        )
    channel_names = ["Red", "Green", "Blue"]

    print("--- Per-channel Residual Statistics ---")

    for index, name in enumerate(channel_names):
        channel = residual[:, :, index]

        print(
            f"{name:>5} | "
            f"mean={channel.mean(dtype=np.float64): .12f} "
            f"std={channel.std(dtype=np.float64): .9f} "
            f"min={channel.min(): .9f} "
            f"max={channel.max(): .9f}"
        )

    # Overall RMS residual
    rms = np.sqrt(
        np.mean(
            residual ** 2,
            dtype=np.float64,
        )
    )
    print(f"Overall residual RMS: {rms:.9f}")


    red_fft = fft_spectrum( residual[:, :, 0] )
    green_fft = fft_spectrum( residual[:, :, 1] )
    blue_fft = fft_spectrum( residual[:, :, 2] )
    all_fft_values = np.concatenate([ red_fft.ravel(), green_fft.ravel(), blue_fft.ravel(), ])
    fft_vmin = np.percentile( all_fft_values, 1.0, )
    fft_vmax = np.percentile( all_fft_values, 99.5, )

    print("--- Shared FFT Display Scale ---")
    print(f"vmin = {fft_vmin:.9f}")
    print(f"vmax = {fft_vmax:.9f}")


    # signed residual visualization
    residual_vis = np.clip( 128.0 + residual * amp_factor, 0, 255, ).astype(np.uint8)

    # flat mean-color reference for visualization
    flat_rgb_uint8 = np.clip( np.round(mean_rgb), 0, 255, ).astype(np.uint8)
    flat_vis = np.empty_like( ai, dtype=np.uint8, )
    flat_vis[:] = flat_rgb_uint8

    # plot
    fig, axes = plt.subplots( 2, 3, figsize=(18, 11), constrained_layout=True, )

    # Original image
    axes[0, 0].set_title( "Original AI image" )

    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")

    # Mean-matched flat reference
    axes[0, 1].set_title( "Mean-matched flat reference" f"Display RGB {tuple(flat_rgb_uint8.tolist())}" )

    axes[0, 1].imshow(flat_vis)
    axes[0, 1].axis("off")


    axes[0, 2].set_title( f"Signed residual ×{amp_factor}" "128 = zero difference" )
    axes[0, 2].imshow( residual_vis )
    axes[0, 2].axis("off")

    fft_image = axes[1, 0].imshow( red_fft, cmap="magma", vmin=fft_vmin, vmax=fft_vmax, ) # red
    axes[1, 0].set_title( "FFT: red residual" )
    axes[1, 0].axis("off")
    axes[1, 1].imshow( green_fft, cmap="magma", vmin=fft_vmin, vmax=fft_vmax, ) # green
    axes[1, 1].set_title( "FFT: green residual" )
    axes[1, 1].axis("off")
    axes[1, 2].imshow( blue_fft, cmap="magma", vmin=fft_vmin, vmax=fft_vmax, )
    axes[1, 2].set_title( "FFT: blue residual" )

    axes[1, 2].axis("off")

    # Shared colorbar
    fig.colorbar( fft_image, ax=axes[1, :], fraction=0.025, pad=0.02, label="log(1 + FFT magnitude)", )

    # All done!!!
    plt.savefig( output_path, dpi=200, bbox_inches="tight", )
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    analyze_zero_mean_residual(
        image_path="ai-generated-images/pure-white-blank-canvas1.png",
        output_path="zero_mean_analysis.png",
        amp_factor=50,
    )