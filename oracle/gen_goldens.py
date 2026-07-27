"""P4 oracle — granular per-sub-op goldens for the FFTformer Swift port.

Doctrine (mlx-porting): generate one .npy per intermediate from the PyTorch oracle so a Swift
parity break localizes to the exact op without needing an MLX-Python twin. Everything is
numpy-seeded, fp32, CPU-torch, and saved C-contiguous in PyTorch NCHW layout — the Swift gate
transposes to NHWC, runs, transposes back, and compares.

Run:  .venv/bin/python gen_goldens.py
Out:  goldens/*.npy  +  goldens/MANIFEST.txt
"""
import os, importlib.util, numpy as np, torch

torch.set_grad_enabled(False)
OUT = "goldens"
os.makedirs(OUT, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "fftformer_arch", "upstream/basicsr/models/archs/fftformer_arch.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

CKPT = "upstream/pretrain_model/fftformer_GoPro.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)

model = A.fftformer()
model.load_state_dict(sd, strict=True)
model.eval()

manifest = []


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(os.path.join(OUT, name + ".npy"), a)
    manifest.append(f"{name+'.npy':46s} {str(a.shape):24s} "
                    f"min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}")
    print(f"  saved {name}.npy  {a.shape}")


def seeded(seed, *shape):
    """Deterministic input, shared by both sides. RNG never crosses the framework boundary."""
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(shape, dtype=np.float32))


def dump(name, t):
    save(name, t.detach().cpu().numpy())


print("\n=== 1. LayerNorm (channel-dim, eps=1e-5, unbiased=False) ===")
# NOTE: BiasFree_LayerNorm is DEAD CODE in this model — TransformerBlock defaults to
# LayerNorm_type='WithBias' and fftformer never overrides it. Only WithBias is ported.
ln = model.encoder_level1[0].norm2
x = seeded(1001, 1, 48, 32, 32)
dump("ln_withbias_in", x)
dump("ln_withbias_out", ln(x))
save("ln_withbias_w", ln.body.weight.numpy())
save("ln_withbias_b", ln.body.bias.numpy())

print("\n=== 2. Patchify / unpatchify (8x8) ===")
from einops import rearrange
xp = seeded(1002, 1, 16, 32, 32)
dump("patch_in", xp)
patched = rearrange(xp, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=8, p2=8)
dump("patch_out", patched)   # (1,16,4,4,8,8)
dump("patch_roundtrip", rearrange(patched, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=8, p2=8))

print("\n=== 3. FFT core: rfft2 -> real-mask multiply -> irfft2 (the DFFN inner op) ===")
# The learned `fft` parameter is REAL and multiplies a COMPLEX spectrum: both the real and
# imaginary parts are scaled by the same gain. That is the single most portable-wrong detail here.
xf = seeded(1003, 1, 16, 4, 4, 8, 8)
mask = seeded(1004, 16, 1, 1, 8, 5)
dump("fftcore_in", xf)
dump("fftcore_mask", mask)
spec = torch.fft.rfft2(xf.float())
dump("fftcore_spec_real", spec.real)
dump("fftcore_spec_imag", spec.imag)
masked = spec * mask
dump("fftcore_out", torch.fft.irfft2(masked, s=(8, 8)))

print("\n=== 4. FSAS core: rfft2(q) * rfft2(k) -> irfft2 (COMPLEX product, not real scaling) ===")
q = seeded(1005, 1, 16, 4, 4, 8, 8)
k = seeded(1006, 1, 16, 4, 4, 8, 8)
dump("fsascore_q", q)
dump("fsascore_k", k)
qf, kf = torch.fft.rfft2(q.float()), torch.fft.rfft2(k.float())
prod = qf * kf
dump("fsascore_prod_real", prod.real)
dump("fsascore_prod_imag", prod.imag)
dump("fsascore_out", torch.fft.irfft2(prod, s=(8, 8)))

print("\n=== 5. DFFN block (real weights, factor 3.0) ===")
dffn = model.encoder_level1[0].ffn
xd = seeded(1007, 1, 48, 32, 32)
dump("dffn_in", xd)
dump("dffn_out", dffn(xd))

print("\n=== 6. FSAS block (real weights, from a decoder block where att=True) ===")
fsas = model.decoder_level1[0].attn
xs = seeded(1008, 1, 48, 32, 32)
dump("fsas_in", xs)
dump("fsas_out", fsas(xs))

print("\n=== 7. TransformerBlock, both modes ===")
tb_no = model.encoder_level1[0]          # att=False
tb_at = model.decoder_level1[0]          # att=True
xt = seeded(1009, 1, 48, 32, 32)
dump("tblock_in", xt)
dump("tblock_noatt_out", tb_no(xt))
dump("tblock_att_out", tb_at(xt))

print("\n=== 8. Down/Up sample (bilinear, align_corners=False) ===")
xu = seeded(1010, 1, 48, 32, 32)
dump("down_in", xu)
dump("down_out", model.down1_2(xu))       # 48 -> 96 ch, /2
xv = seeded(1011, 1, 192, 16, 16)
dump("up_in", xv)
dump("up_out", model.up3_2(xv))           # 192 -> 96 ch, x2

print("\n=== 9. Fuse (its inner block uses ffn factor 2.66, NOT 3 — the trap) ===")
# Named positionally on purpose: `fuse_out == fuse2(fuse_a, fuse_b)`, no argument-order ambiguity
# for the Swift gate to get wrong. (Upstream's own parameter names are `enc, dnc`, and
# fftformer.forward calls it as fuse2(inp_dec_level2, out_enc_level2) — decoder tensor first.)
a = seeded(1012, 1, 96, 32, 32)
b = seeded(1013, 1, 96, 32, 32)
dump("fuse_a", a)
dump("fuse_b", b)
dump("fuse_out", model.fuse2(a, b))

print("\n=== 10. Full model ===")
for size in (64, 128, 256):
    xi = torch.from_numpy(
        np.random.default_rng(2000 + size).random((1, 3, size, size), dtype=np.float32))
    dump(f"full_{size}_in", xi)
    dump(f"full_{size}_out", model(xi))

print("\n=== 11. Realistic image-like input at a production tile (structured, not noise) ===")
g = np.random.default_rng(3001)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
img = np.stack([
    0.5 + 0.4 * np.sin(xx * 12) * np.cos(yy * 9),
    0.5 + 0.4 * np.cos(xx * 7 + yy * 5),
    0.5 + 0.3 * np.sin((xx + yy) * 15),
])[None]
img = np.clip(img + 0.02 * g.standard_normal(img.shape, dtype=np.float32), 0, 1)
xi = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
dump("full_img256_in", xi)
dump("full_img256_out", model(xi))

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write("FFTformer PyTorch goldens — fp32, CPU, PyTorch NCHW layout, C-contiguous.\n")
    f.write(f"checkpoint: {CKPT}\n")
    f.write("constructor: fftformer()  (all defaults; dim=48, num_blocks=[6,6,12,8],\n")
    f.write("             num_refinement_blocks=4, ffn_expansion_factor=3, bias=False)\n")
    f.write("input contract: RGB [0,1], pad to multiple of 32 (reflect, bottom/right), crop back.\n\n")
    f.write("\n".join(manifest) + "\n")

print(f"\n✅ {len(manifest)} goldens written to {OUT}/")
