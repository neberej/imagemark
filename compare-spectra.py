import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


AI_IMAGE = "ai-generated-images/pure-white-blank-canvas3.png"
CONTROL_IMAGE = "local-generated-images/random_control.png"
OUTPUT_IMAGE = "results/spectral_comparison.png"


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    return img, arr


def zero_mean_residual(arr):
    mean_rgb = arr.mean(axis=(0, 1), dtype=np.float64)
    residual = arr - mean_rgb

    residual_means = residual.mean(axis=(0, 1), dtype=np.float64)

    if not np.allclose(residual_means, 0.0, atol=1e-10):
        raise RuntimeError(
            f"Residual is not zero-centered: {residual_means}"
        )

    return residual, mean_rgb


def fft_data(channel):
    """
    Return:
        magnitude: raw FFT magnitude
        log_magnitude: log-scaled FFT magnitude for visualization
        power: FFT power spectrum
    """

    centered = channel - channel.mean(dtype=np.float64)

    h, w = centered.shape

    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    )

    windowed = centered * window

    fft_raw = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_raw)

    magnitude = np.abs(fft_shifted)
    power = magnitude ** 2

    log_magnitude = np.log1p(magnitude)

    return magnitude, log_magnitude, power


def spectral_entropy(power):
    """
    Shannon entropy of normalized spectral power.

    Higher:
        energy is distributed more broadly.

    Lower:
        energy is concentrated into fewer frequency bins.
    """

    flattened = power.ravel().astype(np.float64)

    total = flattened.sum()

    if total <= 0:
        return 0.0

    p = flattened / total

    p = p[p > 0]

    entropy = -np.sum(p * np.log2(p))

    # Normalize by maximum possible entropy so result is 0..1.
    max_entropy = np.log2(flattened.size)

    return entropy / max_entropy


def peak_prominence(power, center_exclusion_radius=10):
    """
    Measure how dominant the strongest non-DC frequency is relative
    to the median spectral power.

    This is a simple exploratory metric, not a watermark detector.
    """

    power_copy = power.copy()

    h, w = power_copy.shape

    cy = h // 2
    cx = w // 2

    y, x = np.ogrid[:h, :w]

    center_mask = (
        (x - cx) ** 2 +
        (y - cy) ** 2
    ) <= center_exclusion_radius ** 2

    power_copy[center_mask] = 0

    max_power = power_copy.max()

    nonzero = power_copy[power_copy > 0]

    if nonzero.size == 0:
        return 0.0

    median_power = np.median(nonzero)

    if median_power <= 0:
        return 0.0

    return max_power / median_power


def radial_power_spectrum(power):
    """
    Collapse 2D power spectrum into average power by radial frequency.
    """

    h, w = power.shape

    cy = h // 2
    cx = w // 2

    y, x = np.indices((h, w))

    radius = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    radius = radius.astype(np.int32)

    radial_sum = np.bincount(
        radius.ravel(),
        weights=power.ravel(),
    )

    radial_count = np.bincount(
        radius.ravel()
    )

    radial_count = np.maximum(
        radial_count,
        1,
    )

    radial_mean = radial_sum / radial_count

    return radial_mean


def spectrum_correlation(log_a, log_b):
    """
    Pearson correlation between two log-magnitude spectra.
    """

    a = log_a.ravel()
    b = log_b.ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return np.corrcoef(a, b)[0, 1]


def print_image_stats(name, arr, mean_rgb, residual):
    print(f"\n--- {name} ---")
    print(f"Shape: {arr.shape}")

    print(
        "Mean RGB: "
        f"R={mean_rgb[0]:.9f}, "
        f"G={mean_rgb[1]:.9f}, "
        f"B={mean_rgb[2]:.9f}"
    )

    print(
        "Residual std: "
        f"R={residual[:, :, 0].std(dtype=np.float64):.9f}, "
        f"G={residual[:, :, 1].std(dtype=np.float64):.9f}, "
        f"B={residual[:, :, 2].std(dtype=np.float64):.9f}"
    )


def analyze_pair(
    ai_path,
    control_path,
    output_path,
):
    ai_img, ai = load_rgb(ai_path)
    control_img, control = load_rgb(control_path)

    if ai.shape != control.shape:
        raise ValueError(
            f"Image shapes differ: "
            f"AI={ai.shape}, control={control.shape}"
        )

    ai_residual, ai_mean = zero_mean_residual(ai)
    control_residual, control_mean = zero_mean_residual(control)

    print_image_stats(
        "AI image",
        ai,
        ai_mean,
        ai_residual,
    )

    print_image_stats(
        "Random control",
        control,
        control_mean,
        control_residual,
    )

    channel_names = [
        "Red",
        "Green",
        "Blue",
    ]

    ai_fft = []
    control_fft = []

    metrics = []

    radial_ai = []
    radial_control = []

    # ------------------------------------------------------------
    # Compute FFTs and metrics
    # ------------------------------------------------------------

    for i, channel_name in enumerate(channel_names):
        ai_mag, ai_log, ai_power = fft_data(
            ai_residual[:, :, i]
        )

        control_mag, control_log, control_power = fft_data(
            control_residual[:, :, i]
        )

        ai_fft.append(ai_log)
        control_fft.append(control_log)

        ai_entropy = spectral_entropy(ai_power)
        control_entropy = spectral_entropy(control_power)

        ai_peak = peak_prominence(ai_power)
        control_peak = peak_prominence(control_power)

        corr = spectrum_correlation(
            ai_log,
            control_log,
        )

        metrics.append({
            "channel": channel_name,
            "ai_entropy": ai_entropy,
            "control_entropy": control_entropy,
            "ai_peak": ai_peak,
            "control_peak": control_peak,
            "correlation": corr,
        })

        radial_ai.append(
            radial_power_spectrum(ai_power)
        )

        radial_control.append(
            radial_power_spectrum(control_power)
        )

    # ------------------------------------------------------------
    # One shared FFT display scale across all six plots
    # ------------------------------------------------------------

    all_fft_values = np.concatenate(
        [
            spectrum.ravel()
            for spectrum
            in ai_fft + control_fft
        ]
    )

    fft_vmin = np.percentile(
        all_fft_values,
        1.0,
    )

    fft_vmax = np.percentile(
        all_fft_values,
        99.5,
    )

    print("\n--- Global FFT Display Scale ---")
    print(f"vmin = {fft_vmin:.9f}")
    print(f"vmax = {fft_vmax:.9f}")

    # ------------------------------------------------------------
    # Print quantitative metrics
    # ------------------------------------------------------------

    print("\n--- Quantitative Spectral Metrics ---")

    print(
        f"{'Channel':<8} "
        f"{'AI entropy':>12} "
        f"{'Ctrl entropy':>13} "
        f"{'AI peak':>14} "
        f"{'Ctrl peak':>14} "
        f"{'Correlation':>13}"
    )

    print("-" * 82)

    for m in metrics:
        print(
            f"{m['channel']:<8} "
            f"{m['ai_entropy']:>12.6f} "
            f"{m['control_entropy']:>13.6f} "
            f"{m['ai_peak']:>14.3f} "
            f"{m['control_peak']:>14.3f} "
            f"{m['correlation']:>13.6f}"
        )

    # ------------------------------------------------------------
    # Figure 1:
    # side-by-side FFT comparison
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
        constrained_layout=True,
    )

    for i, channel_name in enumerate(channel_names):
        fft_image = axes[0, i].imshow(
            ai_fft[i],
            cmap="magma",
            vmin=fft_vmin,
            vmax=fft_vmax,
        )

        axes[0, i].set_title(
            f"AI — {channel_name} FFT"
        )

        axes[0, i].axis("off")

        axes[1, i].imshow(
            control_fft[i],
            cmap="magma",
            vmin=fft_vmin,
            vmax=fft_vmax,
        )

        axes[1, i].set_title(
            f"Control — {channel_name} FFT"
        )

        axes[1, i].axis("off")

    fig.colorbar(
        fft_image,
        ax=axes,
        fraction=0.02,
        pad=0.02,
        label="log(1 + FFT magnitude)",
    )

    fig.suptitle(
        "AI image vs histogram-matched random control",
        fontsize=16,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved FFT comparison: {output_path}")

    # ------------------------------------------------------------
    # Figure 2:
    # radial power comparison
    # ------------------------------------------------------------

    radial_output = (
        output_path.rsplit(".", 1)[0]
        + "_radial.png"
    )

    fig = plt.figure(
        figsize=(12, 7)
    )

    max_radius = min(
        len(radial_ai[0]),
        len(radial_control[0]),
    )

    # Ignore the very center/DC region.
    start_radius = 2

    x = np.arange(
        start_radius,
        max_radius,
    )

    for i, channel_name in enumerate(channel_names):
        ai_curve = radial_ai[i][
            start_radius:max_radius
        ]

        control_curve = radial_control[i][
            start_radius:max_radius
        ]

        # Normalize each curve by total power so shapes are comparable.
        if ai_curve.sum() > 0:
            ai_curve = (
                ai_curve /
                ai_curve.sum()
            )

        if control_curve.sum() > 0:
            control_curve = (
                control_curve /
                control_curve.sum()
            )

        plt.plot(
            x,
            ai_curve,
            label=f"AI {channel_name}",
        )

        plt.plot(
            x,
            control_curve,
            linestyle="--",
            label=f"Control {channel_name}",
        )

    plt.yscale("log")

    plt.xlabel(
        "Radial frequency bin"
    )

    plt.ylabel(
        "Normalized radial power"
    )

    plt.title(
        "Radial power spectrum comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        radial_output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved radial spectrum comparison: "
        f"{radial_output}"
    )



if __name__ == "__main__":
    analyze_pair(
        ai_path=AI_IMAGE,
        control_path=CONTROL_IMAGE,
        output_path=OUTPUT_IMAGE,
    )