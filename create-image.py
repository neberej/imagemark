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
        unique, counts = np.unique(arr[:, :, i], return_counts=True)
        probs = counts / counts.sum()

        print(
            f"{channel_name:>5} | "
            f"mean={channel.mean():.6f} "
            f"std={channel.std():.6f} "
            f"values={list(unique)} "
            f"probs={[round(p, 6) for p in probs]}"
        )


def create_histogram_matched_random_control(
    source_image_path,
    output_path="random_control.png",
    seed=42,
):
    """
    Create a random control image that matches the source image's
    per-channel histogram, but destroys all spatial structure.

    This is a stronger control than hand-picking probabilities.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1. Load source AI image
    # ------------------------------------------------------------

    img = Image.open(source_image_path).convert("RGB")
    source = np.array(img, dtype=np.uint8)

    h, w, c = source.shape

    print("--- Source image loaded ---")
    print(f"Path: {source_image_path}")
    print(f"Dimensions: {w} x {h}")

    summarize_image("Source AI image", source)

    # ------------------------------------------------------------
    # 2. Create a random control with matched per-channel histogram
    # ------------------------------------------------------------

    random_control = np.empty_like(source, dtype=np.uint8)

    channel_names = ["Red", "Green", "Blue"]

    for i, channel_name in enumerate(channel_names):
        channel = source[:, :, i]

        unique_vals, counts = np.unique(channel, return_counts=True)
        probs = counts / counts.sum()

        print(f"\nGenerating random {channel_name} channel...")
        print(f"  unique values: {unique_vals.tolist()}")
        print(f"  probabilities: {[round(float(p), 6) for p in probs]}")

        random_channel = rng.choice(
            unique_vals,
            size=(h, w),
            p=probs,
        ).astype(np.uint8)

        random_control[:, :, i] = random_channel

    # ------------------------------------------------------------
    # 3. Save output
    # ------------------------------------------------------------

    Image.fromarray(random_control).save(output_path)
    print(f"\nSaved random control image to: {output_path}")

    # ------------------------------------------------------------
    # 4. Summarize result
    # ------------------------------------------------------------

    summarize_image("Random histogram-matched control", random_control)


if __name__ == "__main__":
    create_histogram_matched_random_control(
        source_image_path="ai-generated-images/pure-white-blank-canvas3.png",
        output_path="local-generated-images/random_control.png",
        seed=42,
    )