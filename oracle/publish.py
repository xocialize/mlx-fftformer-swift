"""Publish the converted FFTformer weights to mlx-community.

Repo (mlx-community naming: preserve upstream checkpoint identity, quant suffix):
  fftformer_GoPro.pth -> mlx-community/FFTformer-GoPro-fp32

fp32 and not fp16 is deliberate and measured — see the model card and PORT-STATUS.md.

Run:  .venv/bin/python publish.py            (add --dry-run to print the card and stop)
"""
import os
import sys

from huggingface_hub import HfApi

API = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "mlx-community/FFTformer-GoPro-fp32"
WEIGHTS = os.path.join(HERE, "converted", "model-fp32.safetensors")

CARD = """---
library_name: mlx
license: mit
license_link: https://github.com/kkkls/FFTformer/blob/main/license
base_model: kkkls/FFTformer
pipeline_tag: image-to-image
tags:
  - mlx
  - deblurring
  - image-deblurring
  - image-restoration
  - fftformer
---

# mlx-community/FFTformer-GoPro-fp32

[FFTformer](https://github.com/kkkls/FFTformer) motion deblurring (CVPR 2023, `fftformer_GoPro.pth`),
converted to **Apple MLX** for Apple-Silicon inference via the
[`mlx-fftformer-swift`](https://github.com/xocialize/mlx-fftformer-swift) Swift package.

Kong et al., *Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring*.
16,560,474 parameters. Blurry image in → deblurred image at the same resolution out.

The architecture replaces the `QKᵀ` matmul with an element-wise product of two spectra inside each
8×8 patch (`FSAS`), and gates a learned per-frequency mask in the feed-forward path (`DFFN`) — O(N)
in pixels rather than O(N²).

## Use with mlx-fftformer-swift (Swift / MLX)

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-fftformer-swift", from: "0.1.0")
import FFTformerMLXCore

let model = FFTformer()
try model.loadWeights(from: weightsURL)        // model.safetensors from this repo
let deblurred = model.restoreTiled(imageNHWC)  // NHWC RGB in [0,1]
```

Or as an MLXEngine `imageRestore` ModelPackage (`MLXFFTformer.FFTformerRestorePackage`), which
resolves this repo automatically via the Hub.

## Two things to know before you use it

**1. Tile. Do not run this model full-frame.** Level 1 runs at full resolution and both blocks expand
channels 6× (48 → 288), and the patch decomposition copies that again. A single 1080p frame peaks at
about **40 GB** of MLX allocation (measured; predicted within 10% by arithmetic). `restoreTiled`
defaults to a 256-px tile, which holds 1080p to ~8 GB.

⚠️ **Tile geometry must be 32-aligned.** Every FFT block decomposes from the *tile origin*, and a
level-3 patch spans 32 full-res pixels, so a tile origin that is not ≡ 0 (mod 32) shifts the patch
grid out of phase. Measured cost of getting this wrong: ~20.6 dB vs ~26 dB. Both tile size and
overlap must be multiples of 32.

Tiling also matches how the model was trained — GoPro used **128×128 crops**, so it has never seen a
full frame.

**2. This build is fp32, not fp16 — measured, against the usual "small conv nets ship fp16" rule.**
End-to-end vs the fp32 reference on a 256² input:

| dtype | cosine | PSNR vs fp32 |
|---|---|---|
| fp32 | 0.99999994 | 108.99 dB |
| fp16 | 0.99994284 | 50.13 dB |
| bf16 | 0.98789388 | **30.09 dB** |

bf16 is disqualified: 30 dB of *dtype* error is the same order as the model's entire task signal
(GoPro 34.21 dB). fp16 is viable, but at only 66 MB the saving is not worth leaving the dtype error
~16 dB below the signal.

Note that **fp16 beats bf16 here by 20 dB**, inverting the usual default. Peak activation is ~2955,
comfortably inside fp16's range, so nothing overflows — this is mantissa precision, where fp16's 10
bits beat bf16's 7. If you re-quantize, keep the 54 learned `.fft` masks in fp32; the FFT interior is
fp32 by construction upstream (`rfft2(x.float())`).

## Conversion

MLX **NHWC** layout: conv weights `(O,I,kH,kW) → (O,kH,kW,I)` (257 tensors), learned FFT masks
`(C,1,1,8,5) → (8,5,C)` for trailing-dim broadcast (54 tensors), and the `Downsample`/`Upsample`
`nn.Sequential` convs renamed `.body.1. → .conv.` (4 tensors). 535 tensors total, loading
`strict=True` / `verify: .all` clean.

## Parity

Gated against the PyTorch oracle on the CPU stream, fp32, judged on **relative** error:

- **key contract** — 535 tensors, 16,560,474 params, 0 missing / 0 unused / 0 shape mismatches
- **primitives** — 7/7; `patchify`, `unpatchify` and the FFT gate are **bit-identical** (0.00e+00),
  i.e. MLX's `rfft2`/`irfft2` reproduce PyTorch exactly on these 8×8 patches
- **blocks** — 5/5, worst relative 5.4e-07
- **full model** — 64² / 128² / 256² all at cosine **1.00000000**; structured 256² tile 3.6e-05

## Honest limitation

GoPro rank is a weak predictor of real-world value. *"Deblurring in the Wild"* (2026) found that on
real smartphone blur **every** GoPro-trained SOTA method scored *below the blurry input* — baseline
32.38 dB, FFTformer 32.25, the best of a bad set. GoPro's blur is frame-averaged camera shake and so
globally correlated, which is not what handheld capture produces. Validate on your own footage before
relying on it.

Weights: MIT (kkkls/FFTformer, committed in-repo). Port code: MIT.
"""


def main():
    dry = "--dry-run" in sys.argv
    if dry:
        print(CARD)
        print(f"[dry-run] would upload {WEIGHTS} -> {REPO}")
        return

    assert os.path.exists(WEIGHTS), f"missing {WEIGHTS} — run convert.py first"
    size = os.path.getsize(WEIGHTS) / 1e6
    print(f"[publish] {REPO}  ({size:.1f} MB) …")

    API.create_repo(REPO, repo_type="model", exist_ok=True)
    API.upload_file(path_or_fileobj=WEIGHTS, path_in_repo="model.safetensors", repo_id=REPO)
    API.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO)
    print(f"[publish]   ok → https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
