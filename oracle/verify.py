"""P4 Stage 0 verification: does the released checkpoint load strict=True into the default
constructor, and what is the true parameter count / key layout?"""
import sys, torch, importlib.util

# Load the arch module by path — importing it as `basicsr.models.archs...` drags the
# package __init__ chain in, which needs cv2. The arch itself only needs torch + einops.
_spec = importlib.util.spec_from_file_location(
    "fftformer_arch", "upstream/basicsr/models/archs/fftformer_arch.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
fftformer = _m.fftformer

CKPT = "upstream/pretrain_model/fftformer_GoPro.pth"

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
print("=== checkpoint top-level ===")
if isinstance(raw, dict):
    ks = list(raw.keys())
    print(f"  dict with {len(ks)} keys; first 5: {ks[:5]}")
    # BasicSR wraps in 'params'; test.py loads the raw dict.
    if "params" in raw:
        sd = raw["params"]; print("  -> using raw['params']")
    elif "model" in raw:
        sd = raw["model"]; print("  -> using raw['model']")
    else:
        sd = raw; print("  -> using raw dict directly (flat state_dict)")
else:
    sd = raw; print(f"  {type(raw)}")

print(f"\n=== state_dict: {len(sd)} tensors ===")
for k in list(sd.keys())[:6]:
    print(f"  {k:55s} {tuple(sd[k].shape)}  {sd[k].dtype}")
print("  ...")
for k in list(sd.keys())[-4:]:
    print(f"  {k:55s} {tuple(sd[k].shape)}  {sd[k].dtype}")

model = fftformer()  # defaults, exactly as test.py does
n = sum(p.numel() for p in model.parameters())
nb = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"\n=== default-constructed model ===")
print(f"  parameters : {n:,}   ({nb/1e6:.2f} MB fp32 / {nb/2e6:.2f} MB fp16)")

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"\n=== strict=False load ===")
print(f"  missing    : {len(missing)}   {missing[:4]}")
print(f"  unexpected : {len(unexpected)}   {unexpected[:4]}")

try:
    model.load_state_dict(sd, strict=True)
    print("  strict=True: ✅ CLEAN LOAD")
except Exception as e:
    print(f"  strict=True: ❌ {str(e)[:300]}")

# The learned frequency masks — the distinctive parameter of this architecture.
fftp = [(k, tuple(v.shape)) for k, v in sd.items() if k.endswith(".fft")]
print(f"\n=== learned FFT masks: {len(fftp)} ===")
for k, s in fftp[:3]:
    print(f"  {k:55s} {s}")
print(f"  (all rank-5, last dims should be (8, 5) = patch 8x8 rfft2)")

# Confirm the Fuse blocks use the DEFAULT ffn_expansion_factor 2.66, not the model's 3.
print("\n=== ffn_expansion_factor check (the subtle one) ===")
for name in ["encoder_level1.0.ffn.project_in.weight", "fuse1.att_channel.ffn.project_in.weight"]:
    if name in sd:
        o, i = sd[name].shape[:2]
        print(f"  {name:55s} out={o} in={i} -> hidden*2={o}, factor={o/2/i:.4f}")

# Real forward pass on a 32-multiple input, to prove the graph runs end to end.
model.eval()
with torch.no_grad():
    x = torch.rand(1, 3, 64, 64)
    y = model(x)
print(f"\n=== forward smoke ===")
print(f"  in {tuple(x.shape)} -> out {tuple(y.shape)}   range [{y.min():.4f}, {y.max():.4f}]")
print(f"  residual sanity: mean|y-x| = {(y-x).abs().mean():.6f}")
