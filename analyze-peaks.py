import numpy as np
from PIL import Image


IMAGE_PATHS = [
    "ai-generated-images/pure-white-blank-canvas1.png",
    "ai-generated-images/pure-white-blank-canvas2.png",
    "ai-generated-images/pure-white-blank-canvas3.png",
]

NUM_PEAKS = 20
DC_RADIUS = 10
EXCLUSION_RADIUS = 5

# Normalized-frequency tolerance used to decide whether peaks from
# different images represent approximately the same frequency.
# MATCH_TOLERANCE = 0.003 # 3.76 bins
MATCH_TOLERANCE = 1.5 / 1254


def compute_fft_magnitude(channel_data):
    """
    Compute the windowed, zero-mean 2D FFT magnitude.

    Returns:
        magnitude
        image height
        image width
        center y
        center x
    """

    channel_data = channel_data.astype(np.float64)

    # Remove global brightness component.
    centered = channel_data - channel_data.mean(dtype=np.float64)

    h, w = centered.shape

    # Reduce spectral leakage from image boundaries.
    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    )

    windowed = centered * window

    fft_raw = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_raw)

    magnitude = np.abs(fft_shifted)

    cy = h // 2
    cx = w // 2

    return magnitude, h, w, cy, cx


def canonical_frequency(fx, fy):
    """
    Collapse conjugate FFT pairs:

        (+fx, +fy)
        (-fx, -fy)

    into one canonical coordinate.

    Rule:
        Prefer positive fy.
        If fy == 0, prefer positive fx.
    """

    if fy < 0:
        return -fx, -fy

    if np.isclose(fy, 0.0) and fx < 0:
        return -fx, -fy

    return fx, fy


def frequency_distance(a, b):
    """
    Euclidean distance between two normalized frequency coordinates.
    """

    return np.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def get_top_peaks(
    channel_data,
    num_peaks=20,
    dc_radius=10,
    exclusion_radius=5,
):
    """
    Extract distinct FFT peaks using non-maximum suppression.

    Conjugate frequency pairs are collapsed into one canonical
    frequency coordinate.

    Returns:
        list of dictionaries:

        {
            "fx": ...,
            "fy": ...,
            "magnitude": ...,
            "pixel_x": ...,
            "pixel_y": ...
        }
    """

    magnitude, h, w, cy, cx = compute_fft_magnitude(
        channel_data
    )

    mag_copy = magnitude.copy()

    y, x = np.ogrid[:h, :w]

    # ------------------------------------------------------------
    # Remove DC / low-frequency center
    # ------------------------------------------------------------

    dc_mask = (
        (x - cx) ** 2 +
        (y - cy) ** 2
    ) <= dc_radius ** 2

    mag_copy[dc_mask] = 0

    peaks = []

    # Because conjugate pairs will be collapsed, search more raw peaks
    # than we ultimately need.
    raw_search_count = num_peaks * 4

    for _ in range(raw_search_count):

        max_value = mag_copy.max()

        if max_value <= 0:
            break

        flat_index = np.argmax(mag_copy)

        py, px = np.unravel_index(
            flat_index,
            mag_copy.shape,
        )

        fy = (py - cy) / h
        fx = (px - cx) / w

        canonical_fx, canonical_fy = canonical_frequency(
            fx,
            fy,
        )

        # --------------------------------------------------------
        # Check whether this is already represented by its
        # conjugate / nearby equivalent.
        # --------------------------------------------------------

        already_exists = False

        for existing in peaks:
            distance = frequency_distance(
                (canonical_fx, canonical_fy),
                (
                    existing["fx"],
                    existing["fy"],
                ),
            )

            if distance <= (
                exclusion_radius /
                min(h, w)
            ):
                already_exists = True
                break

        if not already_exists:
            peaks.append({
                "fx": canonical_fx,
                "fy": canonical_fy,
                "magnitude": magnitude[py, px],
                "pixel_x": px,
                "pixel_y": py,
            })

        # --------------------------------------------------------
        # Suppress local neighborhood around this raw peak
        # --------------------------------------------------------

        peak_mask = (
            (x - px) ** 2 +
            (y - py) ** 2
        ) <= exclusion_radius ** 2

        mag_copy[peak_mask] = 0

        # --------------------------------------------------------
        # Also suppress conjugate neighborhood.
        #
        # For shifted FFT:
        #
        # conjugate point =
        #     (2*cx - px, 2*cy - py)
        # --------------------------------------------------------

        conjugate_x = 2 * cx - px
        conjugate_y = 2 * cy - py

        if (
            0 <= conjugate_x < w and
            0 <= conjugate_y < h
        ):
            conjugate_mask = (
                (x - conjugate_x) ** 2 +
                (y - conjugate_y) ** 2
            ) <= exclusion_radius ** 2

            mag_copy[conjugate_mask] = 0

        if len(peaks) >= num_peaks:
            break

    return peaks


def find_cross_image_clusters(
    all_image_peaks,
    tolerance=MATCH_TOLERANCE,
):
    """
    Cluster approximately matching frequency peaks across images.

    all_image_peaks:

        [
            peaks_for_image_1,
            peaks_for_image_2,
            ...
        ]

    Returns clusters containing peaks occurring at approximately
    the same normalized frequency.
    """

    clusters = []

    for image_index, peaks in enumerate(all_image_peaks):

        for peak in peaks:

            best_cluster = None
            best_distance = None

            for cluster in clusters:

                center = (
                    cluster["fx"],
                    cluster["fy"],
                )

                distance = frequency_distance(
                    (peak["fx"], peak["fy"]),
                    center,
                )

                if distance <= tolerance:
                    if (
                        best_distance is None or
                        distance < best_distance
                    ):
                        best_cluster = cluster
                        best_distance = distance

            if best_cluster is None:

                clusters.append({
                    "fx": peak["fx"],
                    "fy": peak["fy"],
                    "members": [
                        {
                            "image_index": image_index,
                            **peak,
                        }
                    ],
                })

            else:

                best_cluster["members"].append({
                    "image_index": image_index,
                    **peak,
                })

                # Recalculate cluster center.
                best_cluster["fx"] = np.mean([
                    member["fx"]
                    for member
                    in best_cluster["members"]
                ])

                best_cluster["fy"] = np.mean([
                    member["fy"]
                    for member
                    in best_cluster["members"]
                ])

    return clusters


def print_image_peaks(
    path,
    channel_name,
    peaks,
):
    print(
        f"\n--- {channel_name} Channel Peaks ---"
    )

    print(
        f"{'Rank':<5} | "
        f"{'fx':>10} | "
        f"{'fy':>10} | "
        f"{'Magnitude':>14}"
    )

    print("-" * 50)

    for rank, peak in enumerate(peaks, 1):

        print(
            f"{rank:<5} | "
            f"{peak['fx']:>+10.6f} | "
            f"{peak['fy']:>+10.6f} | "
            f"{peak['magnitude']:>14.2f}"
        )


def print_recurring_clusters(
    clusters,
    num_images,
    channel_name,
):
    """
    Print frequencies that occur in multiple independent images.
    """

    recurring = []

    for cluster in clusters:

        image_ids = {
            member["image_index"]
            for member
            in cluster["members"]
        }

        occurrence_count = len(image_ids)

        if occurrence_count >= 2:

            mean_magnitude = np.mean([
                member["magnitude"]
                for member
                in cluster["members"]
            ])

            recurring.append({
                "fx": cluster["fx"],
                "fy": cluster["fy"],
                "count": occurrence_count,
                "mean_magnitude": mean_magnitude,
                "members": cluster["members"],
            })

    recurring.sort(
        key=lambda item: (
            item["count"],
            item["mean_magnitude"],
        ),
        reverse=True,
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"RECURRING {channel_name.upper()} FREQUENCIES"
    )

    print(
        f"{'=' * 70}"
    )

    if not recurring:
        print(
            "No peaks matched across multiple images "
            f"within tolerance {MATCH_TOLERANCE:.6f}."
        )
        return

    print(
        f"{'fx':>10} | "
        f"{'fy':>10} | "
        f"{'Images':>8} | "
        f"{'Mean magnitude':>16}"
    )

    print("-" * 58)

    for cluster in recurring:

        print(
            f"{cluster['fx']:>+10.6f} | "
            f"{cluster['fy']:>+10.6f} | "
            f"{cluster['count']:>3}/{num_images:<4} | "
            f"{cluster['mean_magnitude']:>16.2f}"
        )


def analyze_cross_image_peaks(
    image_paths,
):
    """
    Extract FFT peaks from multiple images and identify frequencies
    recurring across independent samples.
    """

    channel_names = [
        "Red",
        "Green",
        "Blue",
    ]

    # channel_peaks[channel_index][image_index]
    channel_peaks = [
        [],
        [],
        [],
    ]

    # ------------------------------------------------------------
    # Analyze individual images
    # ------------------------------------------------------------

    for image_index, path in enumerate(image_paths):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"IMAGE {image_index + 1}: {path}"
        )

        print(
            f"{'=' * 70}"
        )

        img = Image.open(path).convert("RGB")

        arr = np.array(
            img,
            dtype=np.float64,
        )

        print(
            f"Dimensions: "
            f"{arr.shape[1]} x {arr.shape[0]}"
        )

        for channel_index, channel_name in enumerate(
            channel_names
        ):

            peaks = get_top_peaks(
                arr[:, :, channel_index],
                num_peaks=NUM_PEAKS,
                dc_radius=DC_RADIUS,
                exclusion_radius=EXCLUSION_RADIUS,
            )

            channel_peaks[
                channel_index
            ].append(peaks)

            print_image_peaks(
                path,
                channel_name,
                peaks,
            )

    # ------------------------------------------------------------
    # Compare peak positions across images
    # ------------------------------------------------------------

    print(
        "\n\n"
        + "#" * 70
    )

    print(
        "CROSS-IMAGE FREQUENCY CONSISTENCY"
    )

    print(
        "#" * 70
    )

    print(
        f"\nImages analyzed: {len(image_paths)}"
    )

    print(
        f"Frequency matching tolerance: "
        f"{MATCH_TOLERANCE:.6f}"
    )

    for channel_index, channel_name in enumerate(
        channel_names
    ):

        clusters = find_cross_image_clusters(
            channel_peaks[channel_index],
            tolerance=MATCH_TOLERANCE,
        )

        print_recurring_clusters(
            clusters,
            len(image_paths),
            channel_name,
        )

if __name__ == "__main__":
    analyze_cross_image_peaks(
        IMAGE_PATHS
    )