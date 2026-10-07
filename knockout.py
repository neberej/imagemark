import numpy as np
from PIL import Image


AI_PATH = "ai-generated-images/pure-white-blank-canvas1.png"
CONTROL_PATH = "local-generated-images/random_control1_exact.png"

TARGET_OUTPUT = "candidate_phase_perturbation.png"
CONTROL_OUTPUT = "matched_control_phase_perturbation.png"

SEED = 42

# Experimental candidate cutoff.
# This is not a SynthID threshold.
CANDIDATE_PERCENTILE = 99.0

# Protect the low-frequency center.
DC_RADIUS = 20

# Strict matching requirements.
MAX_RADIUS_DIFF = 3.0

# Control magnitude must fall between:
# candidate * 0.95 and candidate * 1.05
MAGNITUDE_RATIO_MIN = 0.95
MAGNITUDE_RATIO_MAX = 1.05


def load_rgb_residual(path):
    img = Image.open(path).convert("RGB")

    arr = np.array(
        img,
        dtype=np.float64,
    )

    mean_rgb = arr.mean(
        axis=(0, 1),
        dtype=np.float64,
    )

    residual = arr - mean_rgb

    return arr, residual, mean_rgb


def fft_shifted(channel):
    return np.fft.fftshift(
        np.fft.fft2(channel)
    )


def make_center_mask(h, w, radius):
    cy = h // 2
    cx = w // 2

    y, x = np.ogrid[:h, :w]

    return (
        (x - cx) ** 2
        +
        (y - cy) ** 2
    ) <= radius ** 2


def conjugate_partner(
    y,
    x,
    h,
    w,
):
    """
    Return the Hermitian conjugate partner index
    for an fftshifted spectrum.
    """

    cy = h // 2
    cx = w // 2

    py = (2 * cy - y) % h
    px = (2 * cx - x) % w

    return int(py), int(px)


def canonical_pair(
    y,
    x,
    h,
    w,
):
    """
    Return one canonical representation for a conjugate pair.
    """

    py, px = conjugate_partner(
        y,
        x,
        h,
        w,
    )

    a = (int(y), int(x))
    b = (py, px)

    return min(a, b)


def mask_to_pairs(mask):
    """
    Convert a symmetric mask into unique conjugate pairs.
    """

    h, w = mask.shape

    seen = set()
    pairs = []

    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        key = canonical_pair(
            y,
            x,
            h,
            w,
        )

        if key in seen:
            continue

        seen.add(key)

        py, px = conjugate_partner(
            key[0],
            key[1],
            h,
            w,
        )

        pairs.append(
            (
                key,
                (py, px),
            )
        )

    return pairs


def pairs_to_mask(
    pairs,
    h,
    w,
):
    mask = np.zeros(
        (h, w),
        dtype=bool,
    )

    for a, b in pairs:
        mask[a] = True
        mask[b] = True

    return mask


def calculate_average_power(
    residual,
):
    h, w, c = residual.shape

    power = np.zeros(
        (h, w),
        dtype=np.float64,
    )

    for i in range(c):
        spectrum = fft_shifted(
            residual[:, :, i]
        )

        power += (
            np.abs(spectrum) ** 2
        )

    power /= c

    return power


def calculate_average_magnitude(
    residual,
):
    h, w, c = residual.shape

    magnitude = np.zeros(
        (h, w),
        dtype=np.float64,
    )

    for i in range(c):
        spectrum = fft_shifted(
            residual[:, :, i]
        )

        magnitude += np.abs(
            spectrum
        )

    magnitude /= c

    return magnitude


def radial_distance_map(
    h,
    w,
):
    cy = h // 2
    cx = w // 2

    y, x = np.indices(
        (h, w)
    )

    return np.sqrt(
        (x - cx) ** 2
        +
        (y - cy) ** 2
    )


def build_candidate_mask(
    ai_residual,
    control_residual,
    percentile,
    dc_radius,
):
    h, w, _ = ai_residual.shape

    ai_power = calculate_average_power(
        ai_residual
    )

    control_power = calculate_average_power(
        control_residual
    )

    power_diff = (
        ai_power
        -
        control_power
    )

    center_mask = make_center_mask(
        h,
        w,
        dc_radius,
    )

    valid = power_diff.copy()

    valid[
        center_mask
    ] = 0.0

    positive = valid[
        valid > 0
    ]

    if positive.size == 0:
        raise ValueError(
            "No positive AI-vs-control "
            "power differences found."
        )

    threshold = np.percentile(
        positive,
        percentile,
    )

    raw_mask = (
        valid >= threshold
    )

    # Convert to conjugate pairs and back
    # to guarantee symmetry.
    pairs = mask_to_pairs(
        raw_mask
    )

    candidate_mask = pairs_to_mask(
        pairs,
        h,
        w,
    )

    candidate_mask[
        center_mask
    ] = False

    return (
        candidate_mask,
        ai_power,
        control_power,
        power_diff,
        threshold,
        center_mask,
    )


def get_pair_feature(
    pair,
    radius_map,
    magnitude_map,
):
    """
    Conjugate partners should have the same radius
    and magnitude, so one member is sufficient.
    """

    a, _ = pair

    y, x = a

    return (
        float(
            radius_map[
                y,
                x
            ]
        ),
        float(
            magnitude_map[
                y,
                x
            ]
        ),
    )


def build_strict_matched_masks(
    candidate_mask,
    ai_residual,
    center_mask,
):
    """
    Keep only candidate pairs for which a tightly matched
    non-candidate pair can be found.

    Match requirements:

    - same approximate radial frequency
    - similar original AI FFT magnitude
    - no overlap with candidate region
    - one unique control pair per candidate pair

    Candidates without a valid control are dropped from
    BOTH groups.
    """

    h, w = candidate_mask.shape

    radius_map = radial_distance_map(
        h,
        w,
    )

    magnitude_map = (
        calculate_average_magnitude(
            ai_residual
        )
    )

    original_candidate_pairs = (
        mask_to_pairs(
            candidate_mask
        )
    )

    # Build all possible non-candidate pairs.
    available_mask = (
        ~candidate_mask
        &
        ~center_mask
    )

    all_available_pairs = (
        mask_to_pairs(
            available_mask
        )
    )

    available_pairs = []

    for pair in all_available_pairs:
        a, b = pair

        if (
            candidate_mask[a]
            or candidate_mask[b]
            or center_mask[a]
            or center_mask[b]
        ):
            continue

        available_pairs.append(
            pair
        )

    if not available_pairs:
        raise ValueError(
            "No control pairs available."
        )

    candidate_features = np.array(
        [
            get_pair_feature(
                pair,
                radius_map,
                magnitude_map,
            )
            for pair
            in original_candidate_pairs
        ],
        dtype=np.float64,
    )

    available_features = np.array(
        [
            get_pair_feature(
                pair,
                radius_map,
                magnitude_map,
            )
            for pair
            in available_pairs
        ],
        dtype=np.float64,
    )

    # Track used controls so every candidate
    # gets a unique control pair.
    used_control = np.zeros(
        len(available_pairs),
        dtype=bool,
    )

    accepted_candidate_pairs = []
    accepted_control_pairs = []

    match_stats = []

    # Match hardest/highest magnitude candidates first.
    candidate_order = np.argsort(
        candidate_features[:, 1]
    )[::-1]

    for candidate_index in candidate_order:
        candidate_pair = (
            original_candidate_pairs[
                candidate_index
            ]
        )

        candidate_radius = (
            candidate_features[
                candidate_index,
                0,
            ]
        )

        candidate_magnitude = (
            candidate_features[
                candidate_index,
                1,
            ]
        )

        if candidate_magnitude <= 0:
            continue

        # --------------------------------------------------------
        # Strict candidate filtering
        # --------------------------------------------------------

        unused_indices = np.where(
            ~used_control
        )[0]

        if unused_indices.size == 0:
            break

        control_radius = (
            available_features[
                unused_indices,
                0,
            ]
        )

        control_magnitude = (
            available_features[
                unused_indices,
                1,
            ]
        )

        radius_diff = np.abs(
            control_radius
            -
            candidate_radius
        )

        magnitude_ratio = (
            control_magnitude
            /
            candidate_magnitude
        )

        valid_match = (
            (radius_diff <= MAX_RADIUS_DIFF)
            &
            (
                magnitude_ratio
                >=
                MAGNITUDE_RATIO_MIN
            )
            &
            (
                magnitude_ratio
                <=
                MAGNITUDE_RATIO_MAX
            )
        )

        valid_indices = unused_indices[
            valid_match
        ]

        if valid_indices.size == 0:
            # Drop candidate if no tight match exists.
            continue

        # --------------------------------------------------------
        # Pick best candidate among valid matches.
        #
        # Score:
        #  - radius difference
        #  - distance of magnitude ratio from 1.0
        # --------------------------------------------------------

        valid_radius = (
            available_features[
                valid_indices,
                0,
            ]
        )

        valid_magnitude = (
            available_features[
                valid_indices,
                1,
            ]
        )

        valid_radius_diff = np.abs(
            valid_radius
            -
            candidate_radius
        )

        valid_ratio = (
            valid_magnitude
            /
            candidate_magnitude
        )

        radius_score = (
            valid_radius_diff
            /
            MAX_RADIUS_DIFF
        )

        magnitude_score = (
            np.abs(
                valid_ratio
                -
                1.0
            )
            /
            (
                MAGNITUDE_RATIO_MAX
                -
                1.0
            )
        )

        combined_score = (
            radius_score ** 2
            +
            magnitude_score ** 2
        )

        best_local_index = np.argmin(
            combined_score
        )

        best_control_index = (
            valid_indices[
                best_local_index
            ]
        )

        used_control[
            best_control_index
        ] = True

        control_pair = (
            available_pairs[
                best_control_index
            ]
        )

        matched_radius = (
            available_features[
                best_control_index,
                0,
            ]
        )

        matched_magnitude = (
            available_features[
                best_control_index,
                1,
            ]
        )

        accepted_candidate_pairs.append(
            candidate_pair
        )

        accepted_control_pairs.append(
            control_pair
        )

        match_stats.append(
            {
                "candidate_radius":
                    candidate_radius,

                "control_radius":
                    matched_radius,

                "radius_diff":
                    abs(
                        matched_radius
                        -
                        candidate_radius
                    ),

                "candidate_magnitude":
                    candidate_magnitude,

                "control_magnitude":
                    matched_magnitude,

                "magnitude_ratio":
                    (
                        matched_magnitude
                        /
                        candidate_magnitude
                    ),
            }
        )

    if not accepted_candidate_pairs:
        raise ValueError(
            "No candidate pairs could be "
            "strictly matched."
        )

    final_candidate_mask = pairs_to_mask(
        accepted_candidate_pairs,
        h,
        w,
    )

    final_control_mask = pairs_to_mask(
        accepted_control_pairs,
        h,
        w,
    )

    if np.any(
        final_candidate_mask
        &
        final_control_mask
    ):
        raise RuntimeError(
            "Candidate/control mask overlap detected."
        )

    if (
        final_candidate_mask.sum()
        !=
        final_control_mask.sum()
    ):
        raise RuntimeError(
            "Candidate and control bin counts "
            "do not match."
        )

    return (
        final_candidate_mask,
        final_control_mask,
        original_candidate_pairs,
        accepted_candidate_pairs,
        accepted_control_pairs,
        match_stats,
    )


def generate_random_phase_maps(
    h,
    w,
    channels,
    rng,
):
    """
    Generate one Hermitian-consistent random phase map
    per RGB channel.
    """

    phase_maps = []

    for _ in range(channels):
        noise = rng.standard_normal(
            (h, w)
        )

        noise_fft = np.fft.fftshift(
            np.fft.fft2(noise)
        )

        phase_maps.append(
            np.angle(
                noise_fft
            )
        )

    return phase_maps


def apply_phase_perturbation(
    original_arr,
    residual,
    mean_rgb,
    mask,
    random_phase_maps,
):
    h, w, c = original_arr.shape

    reconstructed = np.zeros_like(
        original_arr
    )

    max_imag_residue = 0.0

    for i in range(c):
        spectrum = fft_shifted(
            residual[:, :, i]
        )

        magnitude = np.abs(
            spectrum
        )

        original_phase = np.angle(
            spectrum
        )

        new_phase = (
            original_phase.copy()
        )

        new_phase[
            mask
        ] = (
            random_phase_maps[
                i
            ][
                mask
            ]
        )

        modified = (
            magnitude
            *
            np.exp(
                1j
                *
                new_phase
            )
        )

        inverse = np.fft.ifft2(
            np.fft.ifftshift(
                modified
            )
        )

        max_imag_residue = max(
            max_imag_residue,
            float(
                np.max(
                    np.abs(
                        inverse.imag
                    )
                )
            ),
        )

        reconstructed[
            :, :, i
        ] = (
            inverse.real
            +
            mean_rgb[i]
        )

    return (
        reconstructed,
        max_imag_residue,
    )


def calculate_metrics(
    original,
    reconstructed,
    affected_bins,
):
    diff_float = (
        reconstructed
        -
        original
    )

    rmse_float = np.sqrt(
        np.mean(
            diff_float ** 2
        )
    )

    mae_float = np.mean(
        np.abs(
            diff_float
        )
    )

    max_abs_float = np.max(
        np.abs(
            diff_float
        )
    )

    original_8bit = np.clip(
        np.round(
            original
        ),
        0,
        255,
    ).astype(
        np.uint8
    )

    reconstructed_8bit = np.clip(
        np.round(
            reconstructed
        ),
        0,
        255,
    ).astype(
        np.uint8
    )

    diff_8bit = (
        original_8bit.astype(
            np.float64
        )
        -
        reconstructed_8bit.astype(
            np.float64
        )
    )

    rmse_8bit = np.sqrt(
        np.mean(
            diff_8bit ** 2
        )
    )

    changed_pixels = np.any(
        original_8bit
        !=
        reconstructed_8bit,
        axis=2,
    )

    changed_subpixels = (
        original_8bit
        !=
        reconstructed_8bit
    )

    return {
        "affected_bins":
            int(
                affected_bins
            ),

        "rmse_float":
            float(
                rmse_float
            ),

        "mae_float":
            float(
                mae_float
            ),

        "max_abs_float":
            float(
                max_abs_float
            ),

        "rmse_8bit":
            float(
                rmse_8bit
            ),

        "pixels_changed_pct":
            float(
                changed_pixels.mean()
                *
                100.0
            ),

        "subpixels_changed_pct":
            float(
                changed_subpixels.mean()
                *
                100.0
            ),

        "output_8bit":
            reconstructed_8bit,
    }


def print_metrics(
    label,
    metrics,
    total_bins,
    max_imag,
):
    fraction = (
        metrics[
            "affected_bins"
        ]
        /
        total_bins
    ) * 100.0

    print(
        f"\n--- {label} ---"
    )

    print(
        f"Affected FFT bins:       "
        f"{metrics['affected_bins']:,} / "
        f"{total_bins:,} "
        f"({fraction:.4f}%)"
    )

    print(
        f"Float RMSE:              "
        f"{metrics['rmse_float']:.6f}"
    )

    print(
        f"Float MAE:               "
        f"{metrics['mae_float']:.6f}"
    )

    print(
        f"Maximum absolute change: "
        f"{metrics['max_abs_float']:.6f}"
    )

    print(
        f"8-bit RMSE:              "
        f"{metrics['rmse_8bit']:.6f}"
    )

    print(
        f"RGB pixels changed:      "
        f"{metrics['pixels_changed_pct']:.4f}%"
    )

    print(
        f"Sub-pixels changed:      "
        f"{metrics['subpixels_changed_pct']:.4f}%"
    )

    print(
        f"Max imaginary residue:   "
        f"{max_imag:.3e}"
    )


def print_match_quality(
    original_candidate_pairs,
    accepted_candidate_pairs,
    accepted_control_pairs,
    match_stats,
):
    print(
        "\n--- Strict Control Match Quality ---"
    )

    print(
        f"Original candidate pairs: "
        f"{len(original_candidate_pairs):,}"
    )

    print(
        f"Accepted candidate pairs: "
        f"{len(accepted_candidate_pairs):,}"
    )

    print(
        f"Matched control pairs:    "
        f"{len(accepted_control_pairs):,}"
    )

    rejected = (
        len(original_candidate_pairs)
        -
        len(accepted_candidate_pairs)
    )

    rejected_pct = (
        rejected
        /
        len(original_candidate_pairs)
        *
        100.0
        if original_candidate_pairs
        else 0.0
    )

    print(
        f"Dropped candidate pairs:  "
        f"{rejected:,} "
        f"({rejected_pct:.2f}%)"
    )

    radius_diffs = np.array(
        [
            item[
                "radius_diff"
            ]
            for item
            in match_stats
        ],
        dtype=np.float64,
    )

    magnitude_ratios = np.array(
        [
            item[
                "magnitude_ratio"
            ]
            for item
            in match_stats
        ],
        dtype=np.float64,
    )

    print(
        "\nRadius matching:"
    )

    print(
        f"Mean difference:          "
        f"{radius_diffs.mean():.4f} bins"
    )

    print(
        f"Median difference:        "
        f"{np.median(radius_diffs):.4f} bins"
    )

    print(
        f"95th percentile:          "
        f"{np.percentile(radius_diffs, 95):.4f} bins"
    )

    print(
        f"Maximum difference:       "
        f"{radius_diffs.max():.4f} bins"
    )

    print(
        "\nMagnitude matching:"
    )

    print(
        f"Mean control/candidate:   "
        f"{magnitude_ratios.mean():.4f}"
    )

    print(
        f"Median control/candidate: "
        f"{np.median(magnitude_ratios):.4f}"
    )

    print(
        f"5th percentile:           "
        f"{np.percentile(magnitude_ratios, 5):.4f}"
    )

    print(
        f"95th percentile:          "
        f"{np.percentile(magnitude_ratios, 95):.4f}"
    )

    print(
        f"Minimum ratio:            "
        f"{magnitude_ratios.min():.4f}"
    )

    print(
        f"Maximum ratio:            "
        f"{magnitude_ratios.max():.4f}"
    )


def run_experiment(
    ai_path,
    control_path,
):
    rng = np.random.default_rng(
        SEED
    )

    (
        ai_arr,
        ai_residual,
        ai_mean,
    ) = load_rgb_residual(
        ai_path
    )

    (
        control_arr,
        control_residual,
        _,
    ) = load_rgb_residual(
        control_path
    )

    if (
        ai_arr.shape
        !=
        control_arr.shape
    ):
        raise ValueError(
            "AI and control images must "
            "have identical dimensions."
        )

    h, w, c = ai_arr.shape

    print(
        "--- Strict Controlled Phase Perturbation ---"
    )

    print(
        f"AI image:         "
        f"{ai_path}"
    )

    print(
        f"Shuffled control: "
        f"{control_path}"
    )

    print(
        f"Candidate percentile: "
        f"{CANDIDATE_PERCENTILE}"
    )

    print(
        f"Maximum radius difference: "
        f"{MAX_RADIUS_DIFF:.2f} bins"
    )

    print(
        "Allowed magnitude ratio: "
        f"{MAGNITUDE_RATIO_MIN:.2f} "
        "to "
        f"{MAGNITUDE_RATIO_MAX:.2f}"
    )

    (
        initial_candidate_mask,
        ai_power,
        control_power,
        power_diff,
        threshold,
        center_mask,
    ) = build_candidate_mask(
        ai_residual,
        control_residual,
        percentile=
            CANDIDATE_PERCENTILE,
        dc_radius=
            DC_RADIUS,
    )

    print(
        "\n--- Initial Candidate Mask ---"
    )

    print(
        f"Power-difference threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Initial candidate bins: "
        f"{initial_candidate_mask.sum():,}"
    )

    (
        candidate_mask,
        control_mask,
        original_candidate_pairs,
        accepted_candidate_pairs,
        accepted_control_pairs,
        match_stats,
    ) = build_strict_matched_masks(
        candidate_mask=
            initial_candidate_mask,
        ai_residual=
            ai_residual,
        center_mask=
            center_mask,
    )

    print(
        f"\nFinal candidate bins: "
        f"{candidate_mask.sum():,}"
    )

    print(
        f"Final control bins:   "
        f"{control_mask.sum():,}"
    )

    print(
        f"Mask overlap:         "
        f"{np.sum(candidate_mask & control_mask)}"
    )

    print_match_quality(
        original_candidate_pairs,
        accepted_candidate_pairs,
        accepted_control_pairs,
        match_stats,
    )

    random_phase_maps = (
        generate_random_phase_maps(
            h,
            w,
            c,
            rng,
        )
    )

    (
        candidate_reconstructed,
        candidate_imag,
    ) = apply_phase_perturbation(
        original_arr=
            ai_arr,
        residual=
            ai_residual,
        mean_rgb=
            ai_mean,
        mask=
            candidate_mask,
        random_phase_maps=
            random_phase_maps,
    )

    (
        control_reconstructed,
        control_imag,
    ) = apply_phase_perturbation(
        original_arr=
            ai_arr,
        residual=
            ai_residual,
        mean_rgb=
            ai_mean,
        mask=
            control_mask,
        random_phase_maps=
            random_phase_maps,
    )

    candidate_metrics = (
        calculate_metrics(
            original=
                ai_arr,
            reconstructed=
                candidate_reconstructed,
            affected_bins=
                candidate_mask.sum(),
        )
    )

    control_metrics = (
        calculate_metrics(
            original=
                ai_arr,
            reconstructed=
                control_reconstructed,
            affected_bins=
                control_mask.sum(),
        )
    )

    total_bins = (
        h * w
    )

    print_metrics(
        "Candidate-region perturbation",
        candidate_metrics,
        total_bins,
        candidate_imag,
    )

    print_metrics(
        "Strictly matched control perturbation",
        control_metrics,
        total_bins,
        control_imag,
    )

    Image.fromarray(
        candidate_metrics[
            "output_8bit"
        ]
    ).save(
        TARGET_OUTPUT
    )

    Image.fromarray(
        control_metrics[
            "output_8bit"
        ]
    ).save(
        CONTROL_OUTPUT
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {TARGET_OUTPUT}"
    )

    print(
        f"  {CONTROL_OUTPUT}"
    )

    print(
        "\n--- Comparison ---"
    )

    if (
        control_metrics[
            "rmse_float"
        ] > 0
    ):
        float_ratio = (
            candidate_metrics[
                "rmse_float"
            ]
            /
            control_metrics[
                "rmse_float"
            ]
        )

        print(
            "Candidate/control float RMSE ratio: "
            f"{float_ratio:.4f}x"
        )

    if (
        control_metrics[
            "rmse_8bit"
        ] > 0
    ):
        bit_ratio = (
            candidate_metrics[
                "rmse_8bit"
            ]
            /
            control_metrics[
                "rmse_8bit"
            ]
        )

        print(
            "Candidate/control 8-bit RMSE ratio: "
            f"{bit_ratio:.4f}x"
        )

    if (
        control_metrics[
            "pixels_changed_pct"
        ] > 0
    ):
        pixel_ratio = (
            candidate_metrics[
                "pixels_changed_pct"
            ]
            /
            control_metrics[
                "pixels_changed_pct"
            ]
        )

        print(
            "Candidate/control changed-pixel ratio: "
            f"{pixel_ratio:.4f}x"
        )


if __name__ == "__main__":
    run_experiment(
        ai_path=AI_PATH,
        control_path=CONTROL_PATH,
    )