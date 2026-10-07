import numpy as np
from PIL import Image


def summarize_image(name, arr):
    arr_f = arr.astype(np.float64)

    print(f"\n--- {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Overall min: {arr.min()}")
    print(f"Overall max: {arr.max()}")
    print(f"Overall mean: {arr_f.mean():.6f}")

    channel_names = ["Red", "Green", "Blue"]

    for i, channel_name in enumerate(channel_names):
        channel = arr[:, :, i].astype(np.float64)

        unique, counts = np.unique(
            arr[:, :, i],
            return_counts=True,
        )

        probs = counts / counts.sum()

        print(
            f"{channel_name:>5} | "
            f"mean={channel.mean():.6f} "
            f"std={channel.std():.6f} "
            f"values={list(unique)} "
            f"probs={[round(float(p), 6) for p in probs]}"
        )


def verify_exact_match(source, control):
    """
    Verify that the shuffled control contains exactly the same
    RGB pixels as the source, only in different spatial positions.
    """

    source_flat = source.reshape(-1, 3)
    control_flat = control.reshape(-1, 3)

    # Encode each RGB pixel into one integer so we can compare
    # the exact joint RGB distribution efficiently.
    source_encoded = (
        source_flat[:, 0].astype(np.uint32) << 16
    ) | (
        source_flat[:, 1].astype(np.uint32) << 8
    ) | (
        source_flat[:, 2].astype(np.uint32)
    )

    control_encoded = (
        control_flat[:, 0].astype(np.uint32) << 16
    ) | (
        control_flat[:, 1].astype(np.uint32) << 8
    ) | (
        control_flat[:, 2].astype(np.uint32)
    )

    source_values, source_counts = np.unique(
        source_encoded,
        return_counts=True,
    )

    control_values, control_counts = np.unique(
        control_encoded,
        return_counts=True,
    )

    same_values = np.array_equal(
        source_values,
        control_values,
    )

    same_counts = np.array_equal(
        source_counts,
        control_counts,
    )

    print("\n--- Exact RGB Distribution Check ---")
    print(f"Same RGB values: {same_values}")
    print(f"Same RGB counts: {same_counts}")

    if not (same_values and same_counts):
        raise RuntimeError(
            "The shuffled control does not preserve "
            "the exact RGB pixel distribution."
        )

    print(
        "Exact joint RGB histogram preserved."
    )


def create_exact_shuffled_control(
    source_image_path,
    output_path="random_control_exact.png",
    seed=42,
):
    """
    Create a control image by shuffling complete RGB pixels.

    This preserves exactly:
    - image dimensions
    - every original RGB pixel
    - the joint RGB distribution
    - each channel histogram
    - each channel mean and standard deviation

    It destroys:
    - spatial arrangement
    - local pixel relationships
    - larger spatial patterns
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1. Load source image
    # ------------------------------------------------------------

    img = Image.open(
        source_image_path
    ).convert("RGB")

    source = np.array(
        img,
        dtype=np.uint8,
    )

    h, w, c = source.shape

    if c != 3:
        raise ValueError(
            f"Expected RGB image, got shape {source.shape}"
        )

    print("--- Source image loaded ---")
    print(f"Path: {source_image_path}")
    print(f"Dimensions: {w} x {h}")

    summarize_image(
        "Source AI image",
        source,
    )

    # ------------------------------------------------------------
    # 2. Flatten complete RGB pixels
    # ------------------------------------------------------------

    flat_pixels = source.reshape(
        -1,
        3,
    ).copy()

    print(
        f"\nTotal RGB pixels: "
        f"{flat_pixels.shape[0]:,}"
    )

    # ------------------------------------------------------------
    # 3. Shuffle pixel positions only
    # ------------------------------------------------------------

    rng.shuffle(
        flat_pixels,
        axis=0,
    )

    shuffled_control = flat_pixels.reshape(
        h,
        w,
        3,
    )

    # ------------------------------------------------------------
    # 4. Save output
    # ------------------------------------------------------------

    Image.fromarray(
        shuffled_control,
        mode="RGB",
    ).save(
        output_path
    )

    print(
        f"\nSaved exact shuffled control to: "
        f"{output_path}"
    )

    # ------------------------------------------------------------
    # 5. Summarize result
    # ------------------------------------------------------------

    summarize_image(
        "Exact shuffled control",
        shuffled_control,
    )

    # ------------------------------------------------------------
    # 6. Verify exact RGB distribution
    # ------------------------------------------------------------

    verify_exact_match(
        source,
        shuffled_control,
    )

    # ------------------------------------------------------------
    # 7. Measure how much spatial position changed
    # ------------------------------------------------------------

    same_position_mask = np.all(
        source == shuffled_control,
        axis=2,
    )

    same_position_count = np.sum(
        same_position_mask
    )

    same_position_pct = (
        same_position_count /
        (h * w)
    ) * 100.0

    print(
        "\n--- Spatial Shuffle Check ---"
    )

    print(
        f"RGB pixels still in matching positions: "
        f"{same_position_count:,} / {h*w:,} "
        f"({same_position_pct:.4f}%)"
    )


if __name__ == "__main__":
    create_exact_shuffled_control(
        source_image_path=(
            "ai-generated-images/"
            "pure-white-blank-canvas1.png"
        ),
        output_path=(
            "local-generated-images/"
            "random_control1_exact.png"
        ),
        seed=42,
    )