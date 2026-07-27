//
//  Ops.swift
//  mlx-fftformer-swift / FFTformerMLXCore
//
//  Role: the spatial + frequency primitives FFTformer needs that MLX doesn't ship
//        directly — 8×8 patch (de)composition and the fp32 FFT island.
//
//  Upstream: https://github.com/kkkls/FFTformer (MIT) — `basicsr/models/archs/fftformer_arch.py`
//  Paper:    Kong et al., "Efficient Frequency Domain-based Transformers for High-Quality
//            Image Deblurring", CVPR 2023.
//
//  Conventions:
//    - NHWC tensor layout (MLX-Swift default; matches the sibling image packages)
//    - Upstream is NCHW and uses einops `rearrange`; the layout mapping is documented per function
//      because getting it wrong is silent (shapes still line up, pixels scramble).
//

import Foundation
import MLX

/// The patch size every FFT block operates on. Fixed at 8 upstream (`self.patch_size = 8` in both
/// `DFFN` and `FSAS`), which is why the learned masks are all `(…, 8, 5)` — `8//2 + 1 = 5`.
public let fftPatchSize = 8

/// Input dimensions must be a multiple of this. Two downsamples put level 3 at H/4, and every
/// level patches by 8, so the binding constraint is 8 × 4 = 32 — not 8.
public let sizeMultiple = 32

// MARK: - Patch decomposition

/// `(B, H, W, C)` → `(B, H/p, W/p, p, p, C)`.
///
/// Upstream (NCHW): `rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2')`.
///
/// In NHWC the channel axis stays last, so the reshape splits H and W in place and a single
/// transpose brings the two patch axes together:
/// `(B, h, p1, w, p2, C)` → `(B, h, w, p1, p2, C)`.
public func patchify(_ x: MLXArray, patch p: Int = fftPatchSize) -> MLXArray {
    precondition(x.ndim == 4, "patchify expects NHWC, got \(x.shape)")
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    precondition(h % p == 0 && w % p == 0, "patchify: \(h)x\(w) not divisible by \(p)")
    return x.reshaped(b, h / p, p, w / p, p, c)
        .transposed(0, 1, 3, 2, 4, 5)
}

/// Inverse of ``patchify(_:patch:)``: `(B, H/p, W/p, p, p, C)` → `(B, H, W, C)`.
public func unpatchify(_ x: MLXArray, patch p: Int = fftPatchSize) -> MLXArray {
    precondition(x.ndim == 6, "unpatchify expects (B,h,w,p,p,C), got \(x.shape)")
    let (b, hh, ww, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(5))
    return x.transposed(0, 1, 3, 2, 4, 5)
        .reshaped(b, hh * p, ww * p, c)
}

// MARK: - The FFT island

/// Axes carrying the two patch dimensions once ``patchify(_:patch:)`` has run.
///
/// In NHWC the patch axes are 3 and 4 (`B, h, w, p1, p2, C`) rather than the trailing pair, so the
/// FFT calls must pass these explicitly. Keeping channels last is deliberate: the alternative —
/// a channels-first patch tensor matching upstream — would transpose activations twice per block,
/// 54 times per forward, to save one transpose of each mask *once* at conversion time.
private let patchAxes = [3, 4]

/// `irfft2(rfft2(x) * mask)` over the patch axes, evaluated in fp32.
///
/// The fp32 island is not a precaution we invented — upstream hardcodes
/// `torch.fft.rfft2(x_patch.float())`, so the spectrum is fp32 there too, and the learned mask is
/// stored fp32 for the same reason. Measured: the model does not survive being run wholesale in
/// bf16 (cosine 0.9839 / 30.1 dB against fp32), so the weights ship fp32 and this stays exact.
///
/// - Parameters:
///   - patched: `(B, h, w, p, p, C)` from ``patchify(_:patch:)``.
///   - mask: `(p, p/2+1, C)` real-valued learned gain. It is REAL and multiplies a COMPLEX
///     spectrum — real and imaginary parts take the same gain. Not a complex mask.
public func fftGate(_ patched: MLXArray, mask: MLXArray, patch p: Int = fftPatchSize) -> MLXArray {
    let spectrum = MLX.rfft2(patched.asType(.float32), axes: patchAxes)
    return MLX.irfft2(spectrum * mask.asType(.float32), s: [p, p], axes: patchAxes)
}

/// `irfft2(rfft2(q) * rfft2(k))` over the patch axes, evaluated in fp32.
///
/// This is FSAS's frequency-domain stand-in for `QKᵀ`: a *complex* product, which by the
/// convolution theorem is circular correlation in the spatial domain. Note it is genuinely
/// `q_fft * k_fft`, not `q_fft * conj(k_fft)` — upstream takes the plain product, so this is a
/// circular *convolution*, and "correcting" it to a correlation would be a port bug.
public func fftMatch(_ q: MLXArray, _ k: MLXArray, patch p: Int = fftPatchSize) -> MLXArray {
    let qf = MLX.rfft2(q.asType(.float32), axes: patchAxes)
    let kf = MLX.rfft2(k.asType(.float32), axes: patchAxes)
    return MLX.irfft2(qf * kf, s: [p, p], axes: patchAxes)
}

// MARK: - Input conditioning

/// Reflect-pads `(B, H, W, C)` up to a multiple of ``sizeMultiple`` on the bottom and right only.
///
/// Mirrors `test.py`:
/// ```python
/// h_n = (32 - h % 32) % 32
/// w_n = (32 - w % 32) % 32
/// input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')
/// ```
/// Bottom/right only — padding symmetrically would shift the output grid against the crop below.
///
/// - Returns: the padded tensor plus the original `(height, width)` to crop back to.
public func padToMultiple(_ x: MLXArray, multiple: Int = sizeMultiple) -> (MLXArray, (Int, Int)) {
    let (h, w) = (x.dim(1), x.dim(2))
    let ph = (multiple - h % multiple) % multiple
    let pw = (multiple - w % multiple) % multiple
    if ph == 0 && pw == 0 { return (x, (h, w)) }

    // MLX has no native reflect pad, and reflection needs the source rows/cols mirrored excluding
    // the edge itself (PyTorch 'reflect', not 'replicate'): for pad p, row H-1-i for i in 1...p.
    var out = x
    if ph > 0 {
        let idx = MLXArray((1...ph).map { Int32(h - 1 - $0) })
        out = concatenated([out, out.take(idx, axis: 1)], axis: 1)
    }
    if pw > 0 {
        let idx = MLXArray((1...pw).map { Int32(w - 1 - $0) })
        out = concatenated([out, out.take(idx, axis: 2)], axis: 2)
    }
    return (out, (h, w))
}

/// Crops back to the pre-pad extent — the `pred[:, :, :h, :w]` in `test.py`.
public func cropTo(_ x: MLXArray, _ size: (Int, Int)) -> MLXArray {
    x[0..., 0 ..< size.0, 0 ..< size.1, 0...]
}
