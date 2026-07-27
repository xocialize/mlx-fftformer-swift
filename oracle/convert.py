"""P4 weight conversion: fftformer_GoPro.pth -> safetensors for MLX-Swift.

Two transforms, and one layout decision that is the crux of the port.

1. Conv weights   PyTorch (O, I, kH, kW) -> MLX (O, kH, kW, I).  Depthwise (C,1,3,3) -> (C,3,3,1),
   which is exactly what MLX Conv2d(groups: C) wants. Same rule, no special case.

2. Learned FFT masks  (C, 1, 1, 8, 5) -> (8, 5, C).
   WHY: in PyTorch NCHW the patch tensor is (B, C, h, w, 8, 8) and rfft2 runs on the trailing
   (p1, p2) axes, so a (C,1,1,8,5) mask broadcasts cleanly. In MLX we keep activations in NHWC to
   avoid transposing them every block: the patch tensor is (B, h, w, 8, 8, C) and we call
   rfft2(x, axes: [3, 4]), giving (B, h, w, 8, 5, C). Trailing-dim broadcasting then wants the mask
   as (8, 5, C). Paying this once at conversion beats transposing activations 54 times per forward.
   (If the Swift side ever moves to a channels-first patch path, transpose back at load — one call.)

The mask is REAL and multiplies a COMPLEX spectrum: real and imaginary parts get the same gain.

Also probes activation magnitude through the FFT path to choose the publish dtype — FFT/FFC nets
are the documented case where fp16 collapses and bf16 is required.

Run:  .venv/bin/python convert.py
"""
import importlib.util, json, os
import numpy as np, torch

torch.set_grad_enabled(False)
OUT = "converted"
os.makedirs(OUT, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "fftformer_arch", "upstream/basicsr/models/archs/fftformer_arch.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

sd = torch.load("upstream/pretrain_model/fftformer_GoPro.pth", map_location="cpu", weights_only=False)

# ---------------------------------------------------------------- transform
converted, stats = {}, {"conv": 0, "fftmask": 0, "vector": 0, "other": 0, "renamed": 0}

# Downsample/Upsample are nn.Sequential(Upsample, Conv2d), so their conv lands at `.body.1.`.
# MLX-Swift module reflection can't reproduce a PyTorch Sequential index for a parameter-free
# first element, so rename to a plain `.conv.`. Only down*/up* match — the LayerNorm wrappers
# are `.body.weight` / `.norm.body.weight` with no numeric index, so they are untouched.
import re
RENAME = re.compile(r"\.body\.1\.")

for k, v in sd.items():
    a = v.detach().cpu().numpy().astype(np.float32)
    if RENAME.search(k):
        k = RENAME.sub(".conv.", k)
        stats["renamed"] += 1
    if k.endswith(".fft"):
        # (C, 1, 1, 8, 5) -> (8, 5, C)
        assert a.ndim == 5 and a.shape[1] == 1 and a.shape[2] == 1, f"{k} {a.shape}"
        a = np.transpose(a[:, 0, 0], (1, 2, 0))
        stats["fftmask"] += 1
    elif a.ndim == 4:
        # conv weight (O, I, kH, kW) -> (O, kH, kW, I)
        a = np.transpose(a, (0, 2, 3, 1))
        stats["conv"] += 1
    elif a.ndim == 1:
        stats["vector"] += 1
    else:
        stats["other"] += 1
    converted[k] = np.ascontiguousarray(a)

print("=== transform summary ===")
for kind, n in stats.items():
    print(f"  {kind:9s}: {n}")
assert stats["other"] == 0, "unexpected tensor rank — inspect before shipping"
print(f"  total    : {len(converted)}")

# --------------------------------------------------- dtype decision probe
print("\n=== activation-magnitude probe (publish-dtype decision) ===")
model = A.fftformer(); model.load_state_dict(sd, strict=True); model.eval()

peak = {"pre_fft": 0.0, "spectrum": 0.0, "post_fft": 0.0, "any": 0.0}
hooks = []


def watch(mod, _inp, out):
    if isinstance(out, torch.Tensor) and out.is_floating_point():
        peak["any"] = max(peak["any"], float(out.abs().max()))


for m in model.modules():
    hooks.append(m.register_forward_hook(watch))

# Instrument one DFFN's FFT interior directly — that is where magnitude actually blows up.
dffn = model.encoder_level1[0].ffn
_orig = dffn.forward


def probed(x):
    from einops import rearrange
    y = dffn.project_in(x)
    peak["pre_fft"] = max(peak["pre_fft"], float(y.abs().max()))
    xp = rearrange(y, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=8, p2=8)
    sp = torch.fft.rfft2(xp.float()) * dffn.fft
    peak["spectrum"] = max(peak["spectrum"], float(sp.abs().max()))
    xp = torch.fft.irfft2(sp, s=(8, 8))
    peak["post_fft"] = max(peak["post_fft"], float(xp.abs().max()))
    y = rearrange(xp, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=8, p2=8)
    x1, x2 = dffn.dwconv(y).chunk(2, dim=1)
    return dffn.project_out(torch.nn.functional.gelu(x1) * x2)


dffn.forward = probed

g = np.random.default_rng(7001)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
img = np.stack([0.5 + 0.4 * np.sin(xx * 12) * np.cos(yy * 9),
                0.5 + 0.4 * np.cos(xx * 7 + yy * 5),
                0.5 + 0.3 * np.sin((xx + yy) * 15)])[None]
img = np.clip(img + 0.02 * g.standard_normal(img.shape, dtype=np.float32), 0, 1)
_ = model(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)))

dffn.forward = _orig
for h in hooks:
    h.remove()

FP16_MAX, BF16_MAX = 65504.0, 3.39e38
for k, v in peak.items():
    print(f"  peak |{k:9s}| = {v:12.2f}")
print(f"\n  fp16 max = {FP16_MAX:,.0f}   bf16 max = {BF16_MAX:.2e}")
headroom = FP16_MAX / max(peak["any"], 1e-9)
print(f"  fp16 headroom vs largest observed activation: {headroom:,.1f}x")
verdict = ("fp16 is SAFE on magnitude grounds (still gate parity at dtype before publishing)"
           if headroom > 50 else
           "fp16 is MARGINAL/UNSAFE — publish bf16 (the documented FFT/FFC failure mode)")
print(f"  => {verdict}")

# ------------------------------------------------------------------ write
try:
    from safetensors.numpy import save_file
except ImportError:
    os.system(".venv/bin/python -m pip install -q safetensors")
    from safetensors.numpy import save_file

meta = {
    "format": "pt",
    "source": "kkkls/FFTformer fftformer_GoPro.pth",
    "license": "MIT",
    "layout": "MLX NHWC; conv (O,kH,kW,I); fft masks (8,5,C)",
    "params": str(sum(int(np.prod(v.shape)) for v in converted.values())),
}
save_file({k: v for k, v in converted.items()}, os.path.join(OUT, "model-fp32.safetensors"), metadata=meta)
save_file({k: v.astype(np.float16) for k, v in converted.items()},
          os.path.join(OUT, "model-fp16.safetensors"), metadata=meta)
# bf16 via ml_dtypes if present, else torch
try:
    import ml_dtypes
    bf = {k: v.astype(ml_dtypes.bfloat16) for k, v in converted.items()}
    save_file(bf, os.path.join(OUT, "model-bf16.safetensors"), metadata=meta)
    bf_ok = True
except Exception as e:
    bf_ok = False
    print(f"\n  (bf16 file skipped: {e})")

print("\n=== written ===")
for f in sorted(os.listdir(OUT)):
    print(f"  {f:32s} {os.path.getsize(os.path.join(OUT, f))/1e6:8.2f} MB")

with open(os.path.join(OUT, "CONVERSION.json"), "w") as f:
    json.dump({"transforms": stats, "peak_activations": peak,
               "fp16_headroom": headroom, "verdict": verdict,
               "params": int(meta["params"]), "bf16_written": bf_ok}, f, indent=2)
print(f"\n  params: {int(meta['params']):,}")
