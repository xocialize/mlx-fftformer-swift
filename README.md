# mlx-fftformer-swift

**FFTformer** motion deblurring on Apple Silicon — an MLX-Swift port, and an MLXEngine
`imageRestore` package.

Upstream: [`kkkls/FFTformer`](https://github.com/kkkls/FFTformer) (MIT code **and** weights,
committed in-repo). Kong et al., *Efficient Frequency Domain-based Transformers for High-Quality
Image Deblurring*, CVPR 2023. 16,560,474 parameters.

Weights: [`mlx-community/FFTformer-GoPro-fp32`](https://huggingface.co/mlx-community/FFTformer-GoPro-fp32).

## Products

| Product | Depends on | Purpose |
|---|---|---|
| `FFTformerMLXCore` | MLX only | the model. Engine-agnostic, usable standalone. |
| `MLXFFTformer` | + MLXToolKit | the MLXEngine `imageRestore` `ModelPackage`. |
| `fftformer-gate` | — | CLI parity / footprint gates (they need a real Metal context). |

It is the **second** package on the existing `imageRestore` capability, alongside NAFNet — selected
by `PackageID`, no contract change.

## Use

```swift
import FFTformerMLXCore

let model = FFTformer()
try model.loadWeights(from: weightsURL)
let deblurred = model.restoreTiled(imageNHWC)   // NHWC RGB in [0,1], same size out
```

## Two things that will bite you

**Tile — do not run this full-frame.** Level 1 runs at full resolution and both blocks expand
channels 6× (48 → 288), and the patch decomposition copies that again. One 1080p frame peaks near
**40 GB** of MLX allocation. `restoreTiled` defaults to a 256-px tile → ~8 GB at 1080p. Because
tiling is internal the peak is one-tile-sized and roughly **flat in resolution** — 4K runs *more*
tiles, not bigger ones.

**Tile geometry must be 32-aligned.** Every FFT block decomposes from the *tile origin*, and a
level-3 patch spans 32 full-res pixels, so a misaligned origin shifts the patch grid out of phase.
Measured cost: ~20.6 dB vs ~26 dB. `restoreTiled` rounds tile and overlap down to a multiple of 32.

Tiling is also *closer to training* than full-frame: GoPro used **128×128 crops**, so the model has
never seen a whole frame.

## fp32, not fp16 — measured

| dtype | cosine | PSNR vs fp32 |
|---|---|---|
| fp32 | 0.99999994 | 108.99 dB |
| fp16 | 0.99994284 | 50.13 dB |
| bf16 | 0.98789388 | **30.09 dB** |

bf16 is disqualified — 30 dB of *dtype* error is the same order as the model's whole task signal
(GoPro 34.21 dB). fp16 is viable; at 66 MB the saving isn't worth the margin. **fp16 beat bf16 by
20 dB**, inverting the usual default: peak activation is ~2955, well inside fp16's range, so this is
mantissa precision (fp16's 10 bits vs bf16's 7), not dynamic range.

## Gates

```bash
swift run fftformer-gate --s0   <weights>              # key contract
swift run fftformer-gate --s1a  <goldens> <weights>    # primitives
swift run fftformer-gate --s2   <goldens> <weights>    # blocks
swift run fftformer-gate --s3   <goldens> <weights>    # full model
swift run fftformer-gate --s4   <goldens> <weights>    # publish-dtype (slow — run alone)
swift run fftformer-gate --tile <weights>              # required-overlap sweep
swift run fftformer-gate --bench <weights>             # split footprint / tile-size sweep
swift test                                             # MAT + CAN + manifest conformance
```

All green: primitives 7/7 (`patchify`, `unpatchify` and the FFT gate **bit-identical**), blocks 5/5,
full model at cosine 1.00000000, conformance 8/8. See [PORT-STATUS.md](PORT-STATUS.md).

## Honest limitation

GoPro rank is a weak predictor of real-world value. *"Deblurring in the Wild"* (2026) found that on
real smartphone blur **every** GoPro-trained SOTA method scored *below the blurry input* — baseline
32.38 dB, FFTformer 32.25, best of a bad set. GoPro's blur is frame-averaged camera shake and
therefore globally correlated, which is not what handheld capture produces.

**This port is parity-verified, not product-validated.** The acceptance study and the capture
protocol it needs are in [REAL-BLUR-VALIDATION.md](REAL-BLUR-VALIDATION.md); it is not yet run.

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

Two FFTformer convs per 256² tile fall inside it: down2_3 (96→192 at H/4, full tiles only) and
up3_2 (192→96 at H/2). `fftformer-gate` pins the CPU stream, so this never showed.

Across most of an image the effect is tiny: relL2 1.25e-4 on a tiled 1024×768 motion-deblur. At the
left image border, though, the network amplifies any perturbation about 100×. There the raw path
reaches 6 levels after tile blending and 14 levels on the bare tile. Winograd is the whole cause:
with `MLX_ENABLE_TF32=0` it is 1 level.

Both sampling convs now take a route (`model.convRoute`, type `FFTformerConvRoute`). **Default
`.conv3d`.**

Measurements: bottom-left 256² tile of that crop, fp32, GPU against the CPU lane.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Output | 8.8e-4 · **max 14 levels** | 1.7e-7 · ≤1 level (truncation) |
| Tile time | 1127 ms | +25 ms |

- Environment override: `FFTFORMER_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- Gate: `FFT_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests`.

## License

Port code MIT. Upstream model and weights MIT (`kkkls`) — see [NOTICE](NOTICE).
