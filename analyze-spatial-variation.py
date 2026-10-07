import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def fft_spectrum(channel):
    """
    Compute a windowed, zero-mean 2D FFT magnitude spectrum.

    Returns:
        2D numpy array containing log(1 + FFT magnitude)
    """

    # Ensure the channel itself is exactly zero-mean.
    centered = channel - channel.mean(dtype=np.float64)

    # Reduce spectral leakage caused by hard image boundaries.
    h, w = centered.shape
    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    )

    windowed = centered * window

    # 2D FFT.
    fft_raw = np.fft.fft2(windowed)

    # Move zero frequency to the center.
    fft_shifted = np.fft.fftshift(fft_raw)

    # Log scale makes lower-energy structure visible.
    magnitude = np.log1p(np.abs(fft_shifted))

    return magnitude


def analyze_zero_mean_residual(
    image_path,
    output_path="zero_mean_analysis.png",
    amp_factor=50,
):
    """
    Analyze spatial variation in an AI-generated near-flat image.

    Method:
        1. Load the image as float64.
        2. Calculate mean RGB independently for each channel.
        3. Treat that mean RGB as the ideal flat reference.
        4. Subtract the mean color from every pixel.
        5. Analyze the resulting zero-mean spatial residual.
        6. Compute FFT spectra for R/G/B independently.
        7. Display all FFTs using one shared color scale.

    Important:
        This reveals spatial structure in the image residual.
        It does not, by itself, prove that the structure is SynthID.
    """

    # ------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------

    img = Image.open(image_path).convert("RGB")

    # float64 is important here.
    #
    # We are measuring extremely small differences around values near 255
    # across more than a million pixels. float32 accumulation can introduce
    # enough numerical error to corrupt the mean and residual.
    ai = np.array(img, dtype=np.float64)

    height, width, channels = ai.shape

    print("--- Image ---")
    print(f"Path: {image_path}")
    print(f"Dimensions: {width} x {height}")
    print(f"Shape: {ai.shape}")

    # ------------------------------------------------------------
    # 2. Raw image statistics
    # ------------------------------------------------------------

    print("\n--- Raw Pixel Statistics ---")
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

    # ------------------------------------------------------------
    # 3. Calculate per-channel mean RGB
    # ------------------------------------------------------------

    mean_rgb = ai.mean(
        axis=(0, 1),
        dtype=np.float64,
    )

    print("\n--- Mean RGB ---")
    print(f"R = {mean_rgb[0]:.9f}")
    print(f"G = {mean_rgb[1]:.9f}")
    print(f"B = {mean_rgb[2]:.9f}")

    # ------------------------------------------------------------
    # 4. Sanity-check the calculated means
    #
    # A channel mean must always fall between that channel's
    # observed minimum and maximum.
    # ------------------------------------------------------------

    channel_min = ai.min(axis=(0, 1))
    channel_max = ai.max(axis=(0, 1))

    print("\n--- Channel Ranges ---")
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
            "Calculated RGB mean is outside the observed pixel range.\n"
            f"Mean: {mean_rgb}\n"
            f"Min:  {channel_min}\n"
            f"Max:  {channel_max}"
        )

    # ------------------------------------------------------------
    # 5. Compute zero-mean residual
    #
    # Conceptually:
    #
    #     residual =
    #         AI image
    #         -
    #         perfectly flat image at mean RGB
    #
    # NumPy broadcasting lets us subtract the three-value RGB mean
    # directly without allocating another full float image.
    # ------------------------------------------------------------

    residual = ai - mean_rgb

    # ------------------------------------------------------------
    # 6. Verify that each residual channel is actually zero-mean
    # ------------------------------------------------------------

    residual_channel_means = residual.mean(
        axis=(0, 1),
        dtype=np.float64,
    )

    print("\n--- Residual ---")
    print(f"Min:  {residual.min():.9f}")
    print(f"Max:  {residual.max():.9f}")
    print(
        f"Overall mean: "
        f"{residual.mean(dtype=np.float64):.12f}"
    )

    print("\n--- Residual Mean By Channel ---")
    print(f"R = {residual_channel_means[0]:.12f}")
    print(f"G = {residual_channel_means[1]:.12f}")
    print(f"B = {residual_channel_means[2]:.12f}")

    if not np.allclose(
        residual_channel_means,
        0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Residual was not centered correctly.\n"
            f"Residual means: {residual_channel_means}"
        )

    # ------------------------------------------------------------
    # 7. Per-channel residual statistics
    # ------------------------------------------------------------

    channel_names = ["Red", "Green", "Blue"]

    print("\n--- Per-channel Residual Statistics ---")

    for index, name in enumerate(channel_names):
        channel = residual[:, :, index]

        print(
            f"{name:>5} | "
            f"mean={channel.mean(dtype=np.float64): .12f} "
            f"std={channel.std(dtype=np.float64): .9f} "
            f"min={channel.min(): .9f} "
            f"max={channel.max(): .9f}"
        )

    # Overall RMS residual across all channels and pixels.
    rms = np.sqrt(
        np.mean(
            residual ** 2,
            dtype=np.float64,
        )
    )

    print(f"\nOverall residual RMS: {rms:.9f}")

    # ------------------------------------------------------------
    # 8. FFT for each channel
    # ------------------------------------------------------------

    red_fft = fft_spectrum(
        residual[:, :, 0]
    )

    green_fft = fft_spectrum(
        residual[:, :, 1]
    )

    blue_fft = fft_spectrum(
        residual[:, :, 2]
    )

    # ------------------------------------------------------------
    # 9. Shared FFT color scale
    #
    # Without a shared scale, matplotlib independently rescales
    # every FFT plot. That can make weak and strong channels look
    # equally intense.
    #
    # Percentiles avoid one extreme peak controlling the whole
    # visualization.
    # ------------------------------------------------------------

    all_fft_values = np.concatenate([
        red_fft.ravel(),
        green_fft.ravel(),
        blue_fft.ravel(),
    ])

    fft_vmin = np.percentile(
        all_fft_values,
        1.0,
    )

    fft_vmax = np.percentile(
        all_fft_values,
        99.5,
    )

    print("\n--- Shared FFT Display Scale ---")
    print(f"vmin = {fft_vmin:.9f}")
    print(f"vmax = {fft_vmax:.9f}")

    # ------------------------------------------------------------
    # 10. Signed residual visualization
    #
    # Zero difference -> 128 gray
    # Positive        -> brighter
    # Negative        -> darker
    #
    # This is only for visualization.
    # The actual residual remains floating-point.
    # ------------------------------------------------------------

    residual_vis = np.clip(
        128.0 + residual * amp_factor,
        0,
        255,
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # 11. Construct flat mean-color reference for visualization
    #
    # The actual mathematical reference is fractional RGB:
    #
    #     [mean_R, mean_G, mean_B]
    #
    # An 8-bit image cannot represent fractional channel values,
    # so this displayed reference is rounded only for visualization.
    # ------------------------------------------------------------

    flat_rgb_uint8 = np.clip(
        np.round(mean_rgb),
        0,
        255,
    ).astype(np.uint8)

    flat_vis = np.empty_like(
        ai,
        dtype=np.uint8,
    )

    flat_vis[:] = flat_rgb_uint8

    # ------------------------------------------------------------
    # 12. Plot
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
        constrained_layout=True,
    )

    # Original image
    axes[0, 0].set_title(
        "Original AI image"
    )

    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")

    # Mean-matched flat reference
    axes[0, 1].set_title(
        "Mean-matched flat reference\n"
        f"Display RGB {tuple(flat_rgb_uint8.tolist())}"
    )

    axes[0, 1].imshow(flat_vis)
    axes[0, 1].axis("off")

    # Signed residual
    axes[0, 2].set_title(
        f"Signed residual ×{amp_factor}\n"
        "128 = zero difference"
    )

    axes[0, 2].imshow(
        residual_vis
    )

    axes[0, 2].axis("off")

    # FFT: Red
    fft_image = axes[1, 0].imshow(
        red_fft,
        cmap="magma",
        vmin=fft_vmin,
        vmax=fft_vmax,
    )

    axes[1, 0].set_title(
        "FFT: red residual"
    )

    axes[1, 0].axis("off")

    # FFT: Green
    axes[1, 1].imshow(
        green_fft,
        cmap="magma",
        vmin=fft_vmin,
        vmax=fft_vmax,
    )

    axes[1, 1].set_title(
        "FFT: green residual"
    )

    axes[1, 1].axis("off")

    # FFT: Blue
    axes[1, 2].imshow(
        blue_fft,
        cmap="magma",
        vmin=fft_vmin,
        vmax=fft_vmax,
    )

    axes[1, 2].set_title(
        "FFT: blue residual"
    )

    axes[1, 2].axis("off")

    # Shared colorbar
    fig.colorbar(
        fft_image,
        ax=axes[1, :],
        fraction=0.025,
        pad=0.02,
        label="log(1 + FFT magnitude)",
    )

    # ------------------------------------------------------------
    # 13. Save
    # ------------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    analyze_zero_mean_residual(
        image_path="random_control.png",
        output_path="zero_mean_analysis.png",
        amp_factor=50,
    )