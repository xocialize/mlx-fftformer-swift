"""Publish-dtype decision, with the mixed-precision contract the architecture actually requires.

FINDING from the naive attempt: you cannot simply do `model.to(float16)`. DFFN/FSAS hardcode
`torch.fft.rfft2(x_patch.float())`, so the spectrum is complex64 while the learned `.fft` mask
would be half -> "expected scalar type Float but found Half".

So the real contract, which the Swift port must mirror:
  * the 54 learned `.fft` masks stay fp32 (a keep_hi_precision set),
  * the FFT itself runs fp32 (upstream forces this already),
  * the result is cast back down before the next conv.
Everything else may be fp16/bf16. This probe measures what that costs.
"""
import importlib.util
import numpy as np, torch
from einops import rearrange

torch.set_grad_enabled(False)
_spec = importlib.util.spec_from_file_location(
    "fftformer_arch", "upstream/basicsr/models/archs/fftformer_arch.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

sd = torch.load("upstream/pretrain_model/fftformer_GoPro.pth", map_location="cpu", weights_only=False)


def build(dt):
    m = A.fftformer()
    m.load_state_dict(sd, strict=True)
    m = m.to(dt).eval()
    if dt is torch.float32:
        return m

    # Mixed-precision contract: masks back to fp32, FFT interior in fp32, cast down after.
    for mod in m.modules():
        if isinstance(mod, A.DFFN):
            mod.fft.data = mod.fft.data.float()

            def dffn_fwd(x, s=mod):
                x = s.project_in(x)
                p = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=s.patch_size, p2=s.patch_size)
                f = torch.fft.rfft2(p.float()) * s.fft
                p = torch.fft.irfft2(f, s=(s.patch_size, s.patch_size))
                x = rearrange(p, 'b c h w p1 p2 -> b c (h p1) (w p2)',
                              p1=s.patch_size, p2=s.patch_size).to(dt)
                x1, x2 = s.dwconv(x).chunk(2, dim=1)
                return s.project_out(torch.nn.functional.gelu(x1) * x2)
            mod.forward = dffn_fwd

        elif isinstance(mod, A.FSAS):
            def fsas_fwd(x, s=mod):
                hidden = s.to_hidden(x)
                q, k, v = s.to_hidden_dw(hidden).chunk(3, dim=1)
                qp = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=s.patch_size, p2=s.patch_size)
                kp = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=s.patch_size, p2=s.patch_size)
                out = torch.fft.irfft2(torch.fft.rfft2(qp.float()) * torch.fft.rfft2(kp.float()),
                                       s=(s.patch_size, s.patch_size))
                out = rearrange(out, 'b c h w p1 p2 -> b c (h p1) (w p2)',
                                p1=s.patch_size, p2=s.patch_size).to(dt)
                return s.project_out(v * s.norm(out))
            mod.forward = fsas_fwd
    return m


g = np.random.default_rng(7001)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
img = np.stack([0.5 + 0.4 * np.sin(xx * 12) * np.cos(yy * 9),
                0.5 + 0.4 * np.cos(xx * 7 + yy * 5),
                0.5 + 0.3 * np.sin((xx + yy) * 15)])[None]
img = np.clip(img + 0.02 * g.standard_normal(img.shape, dtype=np.float32), 0, 1)
x32 = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))

ref = build(torch.float32)(x32)
print("=== end-to-end parity vs fp32 (real weights, 256x256 structured input) ===")

for tag, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    try:
        out = build(dt)(x32.to(dt)).float()
    except Exception as e:
        print(f"  {tag}: FAILED — {str(e)[:120]}")
        continue
    a, b = ref.flatten(), out.flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    mx = float((ref - out).abs().max())
    rel = float((ref - out).abs().mean() / ref.abs().mean())
    psnr = float(10 * torch.log10(1.0 / torch.mean((ref.clamp(0, 1) - out.clamp(0, 1)) ** 2)))
    flag = "OK" if cos >= 0.9999 else ("MARGINAL" if cos >= 0.999 else "FAIL")
    print(f"  {tag}: cos={cos:.8f}  max_abs={mx:.6f}  rel_mean={rel:.2e}  PSNR_vs_fp32={psnr:6.2f} dB  [{flag}]")

print("\n  (PSNR here is fp32-vs-lower-dtype on the OUTPUT IMAGE — for a deterministic")
print("   restoration model this is a meaningful fidelity number, unlike the diffusion case.)")
