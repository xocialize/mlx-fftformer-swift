# Real-blur validation protocol — what footage we need, and why

The queue's acceptance condition for P4 is *"validate on REAL handheld footage, not GoPro."* It comes
from **"Deblurring in the Wild" (2026)**, which found that on real smartphone blur **every**
GoPro-trained SOTA method scored **below the blurry input**: baseline 32.38 dB, NAFNet 31.55,
Restormer 31.89, FFTformer 32.25 — best of a bad set, still a *net loss*.

Parity work (all gates green) proves we ported the model faithfully. It says nothing about whether the
model helps real footage. These are separate questions and this doc is about the second one.

---

## The thing that determines everything: those are **full-reference** numbers

`32.38 dB` is the PSNR of the *blurry input against a sharp ground truth*. The paper's whole finding is
a comparison of two full-reference scores. **Any replication needs a sharp reference for every blurred
frame.** That single requirement drives the entire capture protocol, and it is the reason ordinary
handheld clips — however realistic their blur — cannot answer the question.

### Why no-reference metrics can't substitute

The tempting shortcut is to run a no-reference metric (NIQE, or the SigLIP2 NR-IQA already shipping)
on input vs output and see which scores better. That fails *specifically* for this test: NR-IQA
rewards sharpness and local contrast, so it scores hallucinated detail as an improvement. The failure
mode we are trying to detect — a model inventing plausible-but-wrong texture while losing true detail
— is the one NR metrics are blindest to. NR-IQA is fine as a secondary signal; it cannot be the
decision metric.

---

## Capture protocol: tripod-paired, static scene

The only DIY-feasible way to get true pairs:

1. **Mount the phone** (tripod, clamp, or a genuinely solid surface).
2. **Sharp reference** — fast shutter (≥1/500), lowest usable ISO, good light. This is ground truth.
3. **Blurred capture** — *same scene, unchanged*, camera handheld with deliberate shake, or on the
   mount with a long shutter while you nudge it. **The scene must not change between the two** — no
   moving leaves, no traffic, no people, no shifting light.
4. Repeat across scenes.

**Why a static scene is non-negotiable:** camera motion over a static scene is a pure viewpoint change,
so a homography maps the blurred frame back onto the reference *exactly* for planar or distant subjects.
Any subject motion breaks that and there is no ground truth to recover. This is also why you cannot
simply shoot a moving subject twice — you would need the same instant of motion twice, which is what
beam-splitter rigs (the BSD dataset behind ESTRNN) exist to solve.

### Blackmagic Camera is the right tool for this, and here's the specific reason

Its manual controls are exactly what the protocol needs, and the stock Camera app cannot do:

- **Manual shutter** — force 1/24 or 1/12 to induce genuine *optical* motion blur, rather than hoping
  handheld shake at 1/50 produces enough.
- **Locked ISO and white balance** — the reference and blurred frames must match photometrically, or
  PSNR measures your exposure difference instead of the blur.
- **Less processing** — the stock pipeline applies its own sharpening and noise reduction, so a stock
  frame is already partly "restored"; that shifts the input distribution and muddies the result.

Shoot both halves of each pair with identical manual settings except shutter speed.

---

## How much footage — scene count, not duration

**Scene count is the sample size.** Consecutive video frames are highly correlated, so a 10-second clip
is close to one sample, not 240. Extract **one frame per clip** (or frames several seconds apart).

| Fixture | Count | Purpose |
|---|---|---|
| Blur pairs | **25–40 scenes** | the primary measurement |
| Sharp controls (no induced blur) | **~10 scenes** | prove the model does not *degrade* already-sharp input |
| Alignment noise-floor pairs | **3–5 scenes** | calibrate what is measurable at all (see below) |

Spread the blur pairs across severity (mild / moderate / severe), light level, and content type —
text, faces, foliage, architecture, fine repeating texture. Text and foliage are where hallucination
shows up most legibly.

In wall-clock terms that is one afternoon: ~40 short clips, a few seconds each.

### What that sample size actually buys

Judge on the **paired per-scene delta**, `Δᵢ = PSNR(restored, ref) − PSNR(blurry, ref)`, not on absolute
PSNR. Absolute PSNR varies by several dB across scenes; the paired delta cancels most of that and is far
lower variance.

With a per-scene Δ spread of roughly 0.5–1.5 dB, **n = 25–40 resolves an effect of about ±0.5 dB** —
enough to make a ship / don't-ship call.

**It will not resolve the paper's exact 0.13 dB margin.** That would need several hundred scenes, and
it isn't worth chasing: if the true effect is that small, the honest conclusion is "does not materially
help," which is the same decision either way. Size the study to the decision, not to the paper.

### Measure the noise floor first — this is the step people skip

Before any model runs, shoot **two sharp frames back-to-back on the tripod** and compute PSNR between
them after your alignment pipeline. That number is your ceiling: it bounds any effect you can detect.

If sharp-vs-sharp comes out at, say, 35 dB, then sensor noise plus alignment residual is already
swamping a 0.3 dB effect and the whole study is uninformative until alignment or lighting improves. A
good pipeline on a static scene should land well above 40 dB. **Do this before shooting 40 scenes.**

---

## Analysis

1. Align blurred → reference by homography (feature match, then ECC refinement). Sub-pixel
   misalignment costs several dB and would dwarf the effect being measured.
2. **Crop the border** (~5%) after warping, to drop resampling edge artifacts.
3. Compute PSNR and SSIM for `(blurry, ref)` and `(restored, ref)`.
4. Report mean Δ with a 95% CI, plus the per-scene scatter — a method that wins on average while
   catastrophically failing on text is not a pass.
5. Secondary: NR-IQA delta, to see whether it *disagrees* with the full-reference result. Divergence is
   itself the hallucination signal.

**Pass:** mean Δ > 0 with the CI excluding zero. **The paper's damning result is Δ < 0** — the model
making real footage measurably worse.

---

## Do the clips already provided suffice?

**Partly — they are useful, but they cannot answer the quantitative question.**

| Clip | Verdict |
|---|---|
| `IMG_0161.MOV` (iPhone 17 Pro Max, 1080p HEVC) | Good *qualitative* material and a valid **no-degradation** check. No sharp reference → cannot produce a Δ. |
| `IMG_0007.mov` (2019 iPhone X, Photos export) | Weakest — an export, and its era's pipeline is not what we ship against. |
| `A001_07261300_C001.mov` (Blackmagic Camera, 1214×2160 @ 24 fps) | Confirms the app is installed and its output is readable — the tool for the real fixture set. Still unpaired, so no Δ. |

What they *are* good for right now, at zero extra cost:

- **The no-degradation check.** Run the model over frames from `IMG_0161.MOV` and look for damage on
  already-acceptable footage — the same failure class as P3's luma gate. This needs no reference,
  because the question is "did it get visibly worse," not "how much better."
- **A realistic memory and throughput measurement** at true 1080p (and 1214×2160, which is usefully
  awkward: not a multiple of 32, so it exercises the reflect-pad path).

What they cannot do is tell us whether FFTformer helps or hurts, because there is nothing to score
against. That needs the paired capture above.

---

## Recommended sequence

1. Shoot **3–5 noise-floor pairs** and verify sharp-vs-sharp PSNR is comfortably above 40 dB after
   alignment. Fix the setup before continuing if not.
2. Shoot **25–40 blur pairs + ~10 sharp controls**, Blackmagic Camera, manual shutter, locked ISO/WB.
3. Extract one frame per clip; align; compute Δ.
4. Meanwhile — no new footage needed — run the no-degradation and throughput checks on the clips in hand.
