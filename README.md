# imagemark

A experiment at AI image provenance and watermarking.

Most of the work is based around comparing AI-generated images with locally generated control images.

### Requirements

Most scripts use combination of `numpy`, `Pillow` and `matplotlib`

```
python3 -m venv venv
source venv/bin/activate

pip install numpy pillow matplotlib
```

##### Run as

```
python3 compare-spectra.py
```

## Scripts

| File | Description |
|---|---|
| `analyze-peaks.py` | Finds strong FFT peaks and checks whether similar frequency locations appear across multiple images. |
| `analyze-spatial-variation.py` | Looks at small pixel variations in an image and visualizes their spatial and frequency patterns. |
| `calculate-mean.py` | Calculates basic RGB statistics and residual variation after subtracting the mean color. |
| `compare-spectra.py` | Compares the frequency spectra of AI-generated images with matched control images. |
| `compare-synthid.py` | Compares image properties between SynthID-positive samples and control samples. |
| `create-image.py` | Creates local images used as controls in the experiments. |
| `destroy.py` | Experimental script for testing how image transformations affect detectable provenance signals. |
| `diffusion.py` | Reconstructs an image with a diffusion model and compares how much the regenerated image changes. |
| `fast-fourier-transform.py` | Runs a 2D FFT on image channels and visualizes their frequency-domain structure. |
| `generate-image.py` | Generates test images used throughout the experiments. |
| `knockout.py` | Compares selected FFT regions with closely matched control frequencies using controlled phase changes. |
| `sparse-abalte.py` | Experiments with changing a small set of selected frequency components instead of modifying the whole spectrum. |
| `surgery.py` | Tests targeted changes to selected frequency regions and measures the effect on the reconstructed image. |
| `thresholds-evaulate.py` | Sweeps through FFT magnitude thresholds and measures how each threshold changes the reconstructed image. |
