# mlx-fftformer-swift — port status

**Work order:** P4 in `mlxengine-todo/PORT-QUEUE.md` — FFTformer motion deblur, a **second package on
the existing `imageRestore` capability** (alongside NAFNet's `.goproWidth64`), selected via `PackageID`.
**No engine contract change required.**

Upstream: [`kkkls/FFTformer`](https://github.com/kkkls/FFTformer) · **MIT** (code and the in-repo weights).
Paper: Kong et al., *Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring*, CVPR 2023.

---

## Stage 0 — upstream verification ✅ PASSED (2026-07-26)

| Fact | Verified value |
|---|---|
| License | **MIT**, `Copyright (c) 2023 kkkls` (`LICENSE-upstream`) |
| Weights | `pretrain_model/fftformer_GoPro.pth`, **66,441,171 bytes, committed in-repo** — no Google Drive, no Baidu, no dead academic host |
| Checkpoint shape | **Flat `state_dict`, 535 tensors** — no `params` / `model` wrapper |
| Load | **`strict=True` clean** (0 missing, 0 unexpected) into the *default* constructor |
| Parameters | **16,560,474** — 66.24 MB fp32 |
| Constructor | `fftformer()` verbatim, no overrides: `dim=48`, `num_blocks=[6,6,12,8]`, `num_refinement_blocks=4`, `ffn_expansion_factor=3`, `bias=False` |
| FFT masks | 54 learned tensors, all `(C, 1, 1, 8, 5)` — confirming 8×8 `rfft2` (`8//2+1 = 5`) |

### Input contract (from `test.py`, must be replicated exactly)

RGB in **[0, 1]**; pad to a **multiple of 32** with **`mode='reflect'`, bottom/right only**; run; crop back
`pred[:, :, :h, :w]`; `clamp(0, 1)`. The residual add is **inside** the model (`self.output(...) + inp_img`).

32, not 8: there are two downsamples, so level 3 sits at H/4 and each level patches by 8.

### Traps found by reading source (each would have been a silent failure)

1. **`Fuse` uses a different FFN expansion factor.** `Fuse` constructs `TransformerBlock(dim=n_feat*2)` without
   passing `ffn_expansion_factor`, so it takes the **default 2.66**, not the model's 3. Confirmed in the
   weights: `encoder_level1.0.ffn.project_in` is `288/48` → factor **3.0000**, but
   `fuse1.att_channel.ffn.project_in` is `510/96` → factor **2.6562** (`2 * int(96*2.66) = 510`).
   Passing 3 everywhere yields 576 and fails the key contract.
2. **`num_blocks[3] = 8` is dead config** — the model only reads indices 0, 1, 2.
3. **`BiasFree_LayerNorm` is dead code** — `TransformerBlock` defaults to `LayerNorm_type='WithBias'` and
   `fftformer` never overrides it. Only the WithBias path needs porting.
4. **The learned `.fft` mask is REAL and multiplies a COMPLEX spectrum** — real and imaginary parts take the
   same gain. Not a complex mask.
5. **`Downsample` is `nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)`** — downsampling via
   bilinear interpolate, not strided conv. `align_corners=False` on both directions.

---

## Stage 1 — oracle ✅ COMPLETE

- `oracle/verify.py` — the Stage-0 checks above.
- `oracle/gen_goldens.py` — **39 goldens**, fp32 CPU-torch, numpy-seeded, C-contiguous, **PyTorch NCHW**.
  Committed at `Tests/FFTformerMLXTests/Resources/goldens/`. The Swift gate transposes to NHWC, runs,
  transposes back, compares.
- `oracle/convert.py` — 535 tensors → **257** conv transposes, **54** fft-mask relayouts, **224** vectors
  untouched, **0** unexpected ranks.
- `oracle/dtype_probe.py` — the publish-dtype measurement (below).

### Layout decisions

| Tensor | PyTorch | MLX | Why |
|---|---|---|---|
| Conv weight | `(O, I, kH, kW)` | `(O, kH, kW, I)` | standard; depthwise `(C,1,3,3)` → `(C,3,3,1)` is what `Conv2d(groups: C)` wants — same rule, no special case |
| Learned FFT mask | `(C, 1, 1, 8, 5)` | `(8, 5, C)` | activations stay **NHWC** so nothing is transposed per block. Patch tensor is `(B, h, w, 8, 8, C)`, `rfft2(x, axes: [3, 4])` → `(B, h, w, 8, 5, C)`, and trailing-dim broadcast then wants `(8,5,C)`. Paying this once at conversion beats transposing activations 54× per forward. |

`rfft2` / `irfft2` are in the **main `MLX` module** (`Source/MLX/FFT.swift`) — the `MLXFFT` re-exports are
**deprecated** (*"now available in the main MLX module"*), so this package needs no `MLXFFT` dependency.

### Publish dtype: ship **fp32**; fp16 is viable; **bf16 is disqualified**

Measured **on MLX** (`--s4`), 256² structured input, real weights, against the PyTorch fp32 golden:

| dtype | cosine | rel | PSNR vs fp32 | verdict |
|---|---|---|---|---|
| **fp32** | 0.99999994 | 3.6e-05 | **108.99 dB** | ✅ ships |
| **float16** | 0.99994284 | 2.2e-02 | **50.13 dB** | 🟡 viable, 33 MB |
| **bf16** | 0.98789388 | 4.2e-01 | **30.09 dB** | ❌ disqualified |

⚠️ **This CORRECTS the earlier torch-CPU probe, which reported fp16 as NaN.** That result was flagged at the
time as needing MLX confirmation because torch-CPU half support is flaky — and it was indeed a torch-CPU
artifact. **fp16 does not NaN on MLX.** The bf16 figure, by contrast, reproduced almost exactly across
backends (30.09 dB on MLX vs 30.10 dB on torch), which is what made *it* trustworthy.

**Ship fp32** — not because fp16 fails (it clears the bar) but because the model is only 66 MB, so the 33 MB
saved buys little while a deterministic restoration model wants headroom: at 50 dB the dtype error sits only
~16 dB below the model's own 34.21 dB task signal. Revisit fp16 if size ever matters.

🔑 **Transferable lesson: fp16 beats bf16 here by 20 dB, inverting the usual default.** Peak activation is
~2955 (in `decoder_level2`), comfortably inside fp16's 65504 ceiling, so nothing overflows — this is a
**mantissa-precision** problem, where fp16's 10 mantissa bits beat bf16's 7. The LLM-world preference for
bf16 comes from needing its fp32-equivalent *exponent range*; a small conv/FFT net with bounded activations
has the opposite requirement.

**Mixed-precision contract** for any lower dtype: `model.to(fp16)` is impossible as written —
`DFFN`/`FSAS` hardcode `torch.fft.rfft2(x_patch.float())`, so the spectrum is complex64 while a halved `.fft`
mask throws *"expected scalar type Float but found Half"*. The 54 masks must stay **fp32**, the FFT interior
runs fp32, and the result casts back down before the next conv.

---

## Stage 1 — Swift core ✅ BUILDS · **S0 PASSED**

SPM scaffolded mirroring `mlx-nafnet-swift`: `FFTformerMLXCore` (engine-agnostic, MLX/MLXFast/MLXNN
only) + `MLXFFTformer` (the `imageRestore` wrapper, stubbed) + `fftformer-gate` (CLI gate lane).
**No `MLXFFT` dependency** — `rfft2`/`irfft2` are in the main `MLX` module and the `MLXFFT`
re-exports are deprecated.

```
✅ S0 PASSED — 535 tensors, 16560474 params,
   0 missing / 0 unused / 0 shape mismatches, strict update clean.
```

`swift run fftformer-gate --s0 <weights.safetensors>`. S0 verifies the module tree against the
checkpoint *and* performs a real `update(parameters:verify: .all)` — so it confirms not just that the
key sets match but that the strict verifier accepts the load. It transitively validates every trap
above: the 2.66-vs-3 expansion split, the optional `norm1`/`attn` on encoder blocks, the dead
`reduce_chan_level2` being declared, and the `(8,5,C)` mask relayout.

MLX API notes worth keeping:
- `Upsample(scaleFactor:mode: .linear(alignCorners: false))` is exactly PyTorch's
  `mode='bilinear', align_corners=False`, and takes `scaleFactor: 0.5` for the downsample.
- `MLXNN.gelu` is the erf-exact form `x * (1 + erf(x/√2)) / 2`, matching `F.gelu`'s default.
  Do **not** substitute `geluApproximate`.
- mlx-swift here is **0.31.6**, which carries the NAX split-K GEMM bug (mlx#3797/#3810). FFTformer's
  largest K is ~1152, far below that bug's K ≥ 10240 window — **not affected**.

## Stage 1 — numeric parity ✅ **S1a / S2 / S3 / S4 ALL GREEN**

`swift run fftformer-gate --all <goldens> <weights>`, CPU stream pinned (Apple-GPU fp32 accumulates
~8e-4 relative per op, which both masks and mimics real bugs).

⏱️ **Run S4 separately.** `--s1a`/`--s2`/`--s3` together finish in about a minute, but `--s4`
constructs three full models and runs three 256² forwards on the *CPU* stream and takes several
minutes — enough that `--all` will blow a 10-minute command timeout. Use
`--s1a`/`--s2`/`--s3` for the routine loop and `--s4` only when the dtype question is live.

Gates judge **relative** error (`max_abs / max|ref|`), not absolute. That matters: these tensors span
three orders of magnitude between ops — a LayerNorm output sits near ±2 while a `Fuse` output on
seeded inputs reaches ±2400 — so a single absolute tolerance either fails clean fp32 rounding on the
large tensors or waves through real errors on the small ones. (`fuse` initially "failed" an absolute
1e-4 at max_abs 8.5e-4 while its cosine was 1.00000000 — a gate-design flaw, not a port bug.)

| Gate | Result | Notable |
|---|---|---|
| **S1a** primitives | ✅ 7/7 | `patchify`, `unpatchify`, `fft_gate` are **bit-identical** (rel 0.00e+00) — MLX's `rfft2`/`irfft2` match PyTorch exactly here |
| **S2** blocks | ✅ 5/5 | worst rel 5.4e-07 (`tblock_att`); `fuse` 3.6e-07 |
| **S3** full model | ✅ 4/4 | 64² / 128² / 256² all cosine **1.00000000**; structured 256² tile rel 3.6e-05 |
| **S4** dtype | ✅ ran | see the dtype table above |

The bilinear resamplers carry a 1e-5 tolerance where the other primitives use 1e-6, and that is
evidence-backed rather than goalpost-moving: they are the only primitives computing *interpolation
weights* in floating point and then accumulating a conv over up to 192 channels. The gate's
interior-vs-border diagnostic confirms it — **overall max error equals interior max error (×1.00)**,
i.e. perfectly uniform. A wrong `alignCorners` or an off-by-one sampling grid concentrates error at
the boundary and would appear orders of magnitude larger.

## 🔴 Stage 2 finding: FFTformer **cannot run full-frame** — tiling is mandatory

Measured with `--bench` on an M5 Max / 137 GB:

| input | MLX peak | phys peak |
|---|---|---|
| 512² | 8.70 GB | 33.36 GB |
| 1920×1080 | **39.55 GB** | 109.59 GB |
| 1214×2160 | **50.08 GB** | 107.82 GB |

These are not measurement noise — first-principles arithmetic predicts them within ~10%. Level 1 runs at
**full resolution** and both blocks expand channels **6×** from `dim=48`: `DFFN.project_in` → `hidden*2 =
2·int(48·3) = 288`, `FSAS.to_hidden` → `dim*6 = 288`. At 1080p one 288-channel tensor is **2.41 GB**, and
`patchify` makes a same-size copy. Six encoder blocks holding ~3 live such tensors ⇒ ~43 GB predicted vs
39.55 GB measured. At 1214×2160: 54.9 GB predicted vs 50.08 GB measured.

The 1080p run only "completed" because this machine has 137 GB, and even then at 109 GB phys — deep into
compression/swap. It would fail outright on any normal Mac. **A 66 MB model must not need 40 GB to deblur
a 1080p frame.**

**Tiling is the fix, and it is the natural one here, not a compromise:**

1. **The model was trained on 128×128 crops** — `options/train/GoPro.yml` sets `gt_size: 128`. It has never
   seen a full frame. Tiles are the training distribution.
2. **There is no global receptive field to break.** FSAS is *not* global attention: `rfft2`/`irfft2` run
   **inside each 8×8 patch**, and the q·k product is per-patch. Cross-tile information travels only through
   the 3×3 convs and two downsamples, so the receptive field is finite and modest — which is exactly the
   condition under which crop-valid tiling is *exact*, not approximate.

(NAFNet gets away with full-frame on the same capability because its expansion is 2×, not 6×, and it has no
patch-copy step.)

### 🔑 Tile geometry must be **32-aligned** — measured, and not obvious

The `--tile` overlap sweep (tile 256, vs a full-frame 512² reference) first produced this:

| overlap | 0 | 16 | 32 | 48 | 64 | 96 | 112 |
|---|---|---|---|---|---|---|---|
| PSNR | 26.29 | **20.58** | 25.40 | **20.76** | 26.16 | 26.57 | **20.79** |

No receptive-field trend at all — the bad rows are exactly **16, 48, 112**, i.e. every overlap that
is *not* a multiple of 32. The error tracks `overlap % 32`.

**Why:** each FFT block decomposes its input into 8×8 patches measured from the **tile origin**, and
level 3 runs at ¼ resolution, so one patch there spans **32 full-res pixels**. A tile whose origin is
not ≡ 0 (mod 32) shifts the entire patch grid out of phase with the full-frame decomposition, so the
FFT windows differ *genuinely* — a discretization mismatch no amount of extra context repairs. The
origin is `coreY − overlap`, so **both step and overlap must be multiples of 32**. `restoreTiled` now
rounds both down to a multiple of 32; re-running the sweep confirms 16→0, 48→32, 112→96 produce
byte-identical results to their aligned counterparts, and the 20 dB rows are gone.

### Crop-valid is not achievable — feathered blending instead

Even 32-aligned, tiled-vs-full-frame plateaus around 26–28 dB and barely improves with overlap. That
is the receptive field being genuinely large: 3×3 convs through 6+6+12 encoder and 12+6+6+4
decoder/refinement blocks, many at ¼ resolution, put it in the low hundreds of pixels — an overlap
that big leaves no core. So `restoreTiled` **feathers** (separable linear ramp, image-boundary edges
left unramped) rather than crop-valid hard writes.

**This is not tiling being "wrong."** Full-frame is unattainable at production sizes anyway, and the
model was trained on **128×128 crops** (`gt_size: 128`) — so a 256-px tile is *closer* to its training
distribution than a full frame is. The 26 dB gap is two different valid ways to run a model that has
never seen a full frame, not an error against a privileged reference.

### Tile size: measured trade-off, default 256

1080p, `--bench`, MLX peak / process phys:

| tile | MLX peak | phys | tiles @1080p |
|---|---|---|---|
| 128 | 3.58 GB | 3.51 GB | ~510 |
| 192 | 3.84 GB | 4.32 GB | ~240 |
| **256** ← default | **4.47 GB** | **8.07 GB** | **~60** |
| 384 | 5.82 GB | 10.18 GB | ~30 |

256 is the knee — fits a 16 GB Mac with room while keeping wall-clock sane. ⚠️ A **512** tile costs
8.7 GB MLX / 33 GB phys *per tile* and **SIGTRAPs** partway through a 1080p frame even on a 137 GB
machine; that was the first default and it had to be measured out. Because the package tiles
internally the peak is **one-tile-sized and roughly flat in input resolution** — 4K runs more tiles,
not bigger ones. `tile` is a clean memory lever for a future `BudgetAware` tier.

## Remaining
- [x] ~~Confirm the fp16 NaN on MLX~~ — **done, it was a torch-CPU artifact**; fp16 runs fine on MLX at
      50.13 dB. bf16 independently confirmed bad. See the dtype table.
- [ ] Wrap as a `ModelPackage` on `imageRestore`; MAT / CAN / INF gates born-clean.
- [ ] Publish weights to `mlx-community`; in-app validation + `phys_footprint`; registry row.
- [ ] ⚠️ **Validate on REAL handheld footage, not GoPro.** *"Deblurring in the Wild"* (2026) found every
      GoPro-trained SOTA scored **below the blurry input** on real smartphone blur (baseline 32.38; FFTformer
      32.25 — best of a bad set). GoPro rank is a weak predictor of real-world value.

## Reproduce

```bash
cd oracle && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python torch numpy einops safetensors ml_dtypes
git clone --depth 1 https://github.com/kkkls/FFTformer.git upstream
.venv/bin/python verify.py && .venv/bin/python gen_goldens.py && .venv/bin/python convert.py
```

`upstream/`, `.venv/`, and `converted/` are generated — do not commit. The goldens under
`Tests/.../Resources/` **are** committed (7.9 MB) since they are the parity fixtures.
