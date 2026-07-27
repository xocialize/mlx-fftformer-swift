//
//  FFTformer.swift
//  mlx-fftformer-swift / FFTformerMLXCore
//
//  Role: MLX-Swift port of FFTformer — frequency-domain transformer for motion deblurring.
//        Ships as a SECOND package on the engine's existing `imageRestore` capability, selected
//        by `PackageID` alongside NAFNet's `.goproWidth64`. No contract change.
//
//  Upstream: https://github.com/kkkls/FFTformer (MIT, weights committed in-repo)
//  Paper:    Kong et al., "Efficient Frequency Domain-based Transformers for High-Quality
//            Image Deblurring", CVPR 2023.
//
//  Conventions:
//    - NHWC layout; module property keys mirror the upstream state_dict exactly so weights load
//      with `.noUnusedKeys` (see PORT-STATUS.md for the one rename: `.body.1.` → `.conv.`)
//    - `@unchecked Sendable` for classes holding MLX state (sibling convention)
//    - fp32 throughout — measured, not assumed (PORT-STATUS.md "Publish dtype")
//
//  Structure is kept isomorphic to `fftformer_arch.py`: same class names, same decomposition,
//  same forward order. Diffing the two files should show only PyTorch↔MLX op substitutions.
//

import Foundation
import MLX
import MLXNN

// MARK: - Normalization

/// Channel-dimension LayerNorm, `WithBias` variant.
///
/// Upstream normalizes over the channel axis after `to_3d` (`b c h w -> b (h w) c`) with
/// `var(unbiased=False)` and a hardcoded `eps = 1e-5`. In NHWC the channel axis is already last,
/// so `MLXNN.LayerNorm` is the identical computation and no reshaping is needed at all.
///
/// The nested `body` property exists only so the key path matches upstream's
/// `LayerNorm { body: WithBias_LayerNorm }` wrapper — `norm2.body.weight`.
///
/// Upstream's `BiasFree_LayerNorm` is dead code: `TransformerBlock` defaults to
/// `LayerNorm_type='WithBias'` and `fftformer` never overrides it. Not ported.
public final class ChannelLayerNorm: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "body") public var body: LayerNorm

    public init(dim: Int) {
        self._body.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { body(x) }
}

// MARK: - DFFN — the frequency-gated feed-forward block

/// Discriminative Frequency-domain Feed-Forward Network.
///
/// A learned per-frequency gain is applied inside each 8×8 patch, then a depthwise conv and a
/// gated-GELU. `ffnExpansionFactor` is 3 for every block the model builds directly — but **2.66**
/// inside `Fuse`, which constructs its `TransformerBlock` without passing the factor and so takes
/// the constructor default. That is not a typo: it is visible in the released weights
/// (`fuse1.att_channel.ffn.project_in` is 510×96 = `2 * int(96 * 2.66)`), and passing 3 uniformly
/// yields 576 channels and fails the key contract.
public final class DFFN: Module, UnaryLayer, @unchecked Sendable {
    /// Learned real-valued frequency gain, stored `(patch, patch/2+1, C)` for NHWC trailing-dim
    /// broadcast against the `(B, h, w, p, p/2+1, C)` spectrum. Upstream stores `(C, 1, 1, p, p/2+1)`.
    @ParameterInfo(key: "fft") public var fft: MLXArray

    @ModuleInfo(key: "project_in") public var projectIn: Conv2d
    @ModuleInfo(key: "dwconv") public var dwconv: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d

    public init(dim: Int, ffnExpansionFactor: Float, bias: Bool) {
        let hidden = Int(Float(dim) * ffnExpansionFactor)   // int() truncation — 2.66 → 255, not 256
        self._fft.wrappedValue = MLXArray.ones([fftPatchSize, fftPatchSize / 2 + 1, hidden * 2])
        self._projectIn.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: hidden * 2, kernelSize: 1, bias: bias)
        self._dwconv.wrappedValue = Conv2d(
            inputChannels: hidden * 2, outputChannels: hidden * 2, kernelSize: 3,
            padding: 1, groups: hidden * 2, bias: bias)
        self._projectOut.wrappedValue = Conv2d(
            inputChannels: hidden, outputChannels: dim, kernelSize: 1, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        var y = projectIn(x)
        y = unpatchify(fftGate(patchify(y), mask: fft)).asType(dtype)
        let parts = dwconv(y).split(parts: 2, axis: -1)
        return projectOut(gelu(parts[0]) * parts[1])
    }
}

// MARK: - FSAS — frequency self-attention

/// Frequency domain-based Self-Attention Solver.
///
/// Replaces the `QKᵀ` matmul with an element-wise product of the two spectra inside each 8×8
/// patch — O(N) in pixels rather than O(N²) — then normalizes and modulates `V`.
public final class FSAS: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "to_hidden") public var toHidden: Conv2d
    @ModuleInfo(key: "to_hidden_dw") public var toHiddenDW: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d
    @ModuleInfo(key: "norm") public var norm: ChannelLayerNorm

    public init(dim: Int, bias: Bool) {
        self._toHidden.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim * 6, kernelSize: 1, bias: bias)
        self._toHiddenDW.wrappedValue = Conv2d(
            inputChannels: dim * 6, outputChannels: dim * 6, kernelSize: 3,
            padding: 1, groups: dim * 6, bias: bias)
        self._projectOut.wrappedValue = Conv2d(
            inputChannels: dim * 2, outputChannels: dim, kernelSize: 1, bias: bias)
        self._norm.wrappedValue = ChannelLayerNorm(dim: dim * 2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let qkv = toHiddenDW(toHidden(x)).split(parts: 3, axis: -1)
        let matched = fftMatch(patchify(qkv[0]), patchify(qkv[1]))
        let out = norm(unpatchify(matched).asType(dtype))
        return projectOut(qkv[2] * out)
    }
}

// MARK: - Transformer block

/// `att == false` in the encoders (FFN only); `att == true` in every decoder and the refinement
/// stage. When `att` is false, `norm1`/`attn` are not constructed at all — which is why the
/// encoder keys carry only `norm2` and `ffn`.
public final class TransformerBlock: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "norm1") public var norm1: ChannelLayerNorm?
    @ModuleInfo(key: "attn") public var attn: FSAS?
    @ModuleInfo(key: "norm2") public var norm2: ChannelLayerNorm
    @ModuleInfo(key: "ffn") public var ffn: DFFN

    public init(dim: Int, ffnExpansionFactor: Float = 2.66, bias: Bool = false, att: Bool = false) {
        if att {
            self._norm1.wrappedValue = ChannelLayerNorm(dim: dim)
            self._attn.wrappedValue = FSAS(dim: dim, bias: bias)
        } else {
            self._norm1.wrappedValue = nil
            self._attn.wrappedValue = nil
        }
        self._norm2.wrappedValue = ChannelLayerNorm(dim: dim)
        self._ffn.wrappedValue = DFFN(dim: dim, ffnExpansionFactor: ffnExpansionFactor, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if let attn, let norm1 { y = y + attn(norm1(y)) }
        return y + ffn(norm2(y))
    }
}

// MARK: - Resampling

/// `nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), Conv2d)`.
///
/// Note this downsamples by *bilinear interpolation*, not a strided conv — an easy thing to
/// "improve" into a stride-2 conv and silently break.
public final class Downsample: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    private let resample = Upsample(scaleFactor: 0.5, mode: .linear(alignCorners: false))

    public init(dim: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim * 2, kernelSize: 3, padding: 1, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv(resample(x)) }
}

/// `nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Conv2d)`.
public final class Upsample2x: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    private let resample = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))

    public init(dim: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv(resample(x)) }
}

// MARK: - Fuse

/// Skip-connection fusion: concat, 1×1 conv, a transformer block, 1×1 conv, then split and ADD
/// the halves (not concat) so the channel count returns to `nFeat`.
///
/// Its inner `TransformerBlock` takes the DEFAULT `ffnExpansionFactor` of 2.66 — see ``DFFN``.
public final class Fuse: Module, @unchecked Sendable {
    @ModuleInfo(key: "att_channel") public var attChannel: TransformerBlock
    @ModuleInfo(key: "conv") public var conv: Conv2d
    @ModuleInfo(key: "conv2") public var conv2: Conv2d
    private let nFeat: Int

    public init(nFeat: Int) {
        self.nFeat = nFeat
        self._attChannel.wrappedValue = TransformerBlock(dim: nFeat * 2)
        self._conv.wrappedValue = Conv2d(
            inputChannels: nFeat * 2, outputChannels: nFeat * 2, kernelSize: 1, bias: true)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: nFeat * 2, outputChannels: nFeat * 2, kernelSize: 1, bias: true)
    }

    /// Argument order matches upstream `forward(self, enc, dnc)`; note `fftformer.forward` calls it
    /// as `fuse2(inp_dec_level2, out_enc_level2)` — decoder tensor first.
    public func callAsFunction(_ enc: MLXArray, _ dnc: MLXArray) -> MLXArray {
        let x = conv2(attChannel(conv(concatenated([enc, dnc], axis: -1))))
        let halves = x.split(parts: 2, axis: -1)
        return halves[0] + halves[1]
    }
}

// MARK: - Patch embedding

public final class OverlapPatchEmbed: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "proj") public var proj: Conv2d

    public init(inChannels: Int = 3, embedDim: Int = 48, bias: Bool = false) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: embedDim,
            kernelSize: 3, padding: 1, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

// MARK: - The model

public final class FFTformer: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var inpChannels = 3
        public var outChannels = 3
        public var dim = 48
        /// Index 3 (`8`) is DEAD — upstream only ever reads indices 0, 1, 2. Kept so the literal
        /// matches `fftformer_arch.py` and nobody "fixes" it back later.
        public var numBlocks = [6, 6, 12, 8]
        public var numRefinementBlocks = 4
        public var ffnExpansionFactor: Float = 3
        public var bias = false

        /// The released `fftformer_GoPro.pth` uses every default — `test.py` calls `fftformer()`
        /// with no arguments.
        public init() {}
    }

    @ModuleInfo(key: "patch_embed") public var patchEmbed: OverlapPatchEmbed
    @ModuleInfo(key: "encoder_level1") public var encoderLevel1: [TransformerBlock]
    @ModuleInfo(key: "down1_2") public var down1_2: Downsample
    @ModuleInfo(key: "encoder_level2") public var encoderLevel2: [TransformerBlock]
    @ModuleInfo(key: "down2_3") public var down2_3: Downsample
    @ModuleInfo(key: "encoder_level3") public var encoderLevel3: [TransformerBlock]
    @ModuleInfo(key: "decoder_level3") public var decoderLevel3: [TransformerBlock]
    @ModuleInfo(key: "up3_2") public var up3_2: Upsample2x
    /// DEAD MODULE — constructed upstream, present in the checkpoint, never called in `forward`.
    /// Declared so weight loading stays `.noUnusedKeys` clean.
    @ModuleInfo(key: "reduce_chan_level2") public var reduceChanLevel2: Conv2d
    @ModuleInfo(key: "decoder_level2") public var decoderLevel2: [TransformerBlock]
    @ModuleInfo(key: "up2_1") public var up2_1: Upsample2x
    @ModuleInfo(key: "decoder_level1") public var decoderLevel1: [TransformerBlock]
    @ModuleInfo(key: "refinement") public var refinement: [TransformerBlock]
    @ModuleInfo(key: "fuse2") public var fuse2: Fuse
    @ModuleInfo(key: "fuse1") public var fuse1: Fuse
    @ModuleInfo(key: "output") public var output: Conv2d

    public init(_ cfg: Configuration = Configuration()) {
        let d = cfg.dim
        let f = cfg.ffnExpansionFactor
        let b = cfg.bias

        func blocks(_ n: Int, _ dim: Int, att: Bool) -> [TransformerBlock] {
            (0 ..< n).map { _ in
                TransformerBlock(dim: dim, ffnExpansionFactor: f, bias: b, att: att)
            }
        }

        self._patchEmbed.wrappedValue = OverlapPatchEmbed(
            inChannels: cfg.inpChannels, embedDim: d, bias: b)

        self._encoderLevel1.wrappedValue = blocks(cfg.numBlocks[0], d, att: false)
        self._down1_2.wrappedValue = Downsample(dim: d)
        self._encoderLevel2.wrappedValue = blocks(cfg.numBlocks[1], d * 2, att: false)
        self._down2_3.wrappedValue = Downsample(dim: d * 2)
        self._encoderLevel3.wrappedValue = blocks(cfg.numBlocks[2], d * 4, att: false)

        self._decoderLevel3.wrappedValue = blocks(cfg.numBlocks[2], d * 4, att: true)
        self._up3_2.wrappedValue = Upsample2x(dim: d * 4)
        self._reduceChanLevel2.wrappedValue = Conv2d(
            inputChannels: d * 4, outputChannels: d * 2, kernelSize: 1, bias: b)
        self._decoderLevel2.wrappedValue = blocks(cfg.numBlocks[1], d * 2, att: true)
        self._up2_1.wrappedValue = Upsample2x(dim: d * 2)
        self._decoderLevel1.wrappedValue = blocks(cfg.numBlocks[0], d, att: true)
        self._refinement.wrappedValue = blocks(cfg.numRefinementBlocks, d, att: true)

        self._fuse2.wrappedValue = Fuse(nFeat: d * 2)
        self._fuse1.wrappedValue = Fuse(nFeat: d)
        self._output.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: cfg.outChannels,
            kernelSize: 3, padding: 1, bias: b)
    }

    /// Forward on an already-padded NHWC tensor whose H and W are multiples of ``sizeMultiple``.
    /// Use ``restore(_:)`` for the padding/cropping contract.
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let e1 = encoderLevel1.reduce(patchEmbed(input)) { $1($0) }
        let e2 = encoderLevel2.reduce(down1_2(e1)) { $1($0) }
        let e3 = encoderLevel3.reduce(down2_3(e2)) { $1($0) }

        let d3 = decoderLevel3.reduce(e3) { $1($0) }

        // Upstream applies `reduce_chan_level2` nowhere — the fuse handles the channel math.
        let d2 = decoderLevel2.reduce(fuse2(up3_2(d3), e2)) { $1($0) }
        let d1 = decoderLevel1.reduce(fuse1(up2_1(d2), e1)) { $1($0) }
        let refined = refinement.reduce(d1) { $1($0) }

        return output(refined) + input   // residual is inside the model
    }

    /// The full inference contract from `test.py`: reflect-pad to a multiple of 32, run, crop back,
    /// clamp to [0, 1]. Input and output are NHWC RGB in [0, 1].
    public func restore(_ image: MLXArray) -> MLXArray {
        let (padded, size) = padToMultiple(image)
        return clip(cropTo(self(padded), size), min: 0, max: 1)
    }

    /// Tiled restoration — **the production path.** Full-frame is not viable above ~512².
    ///
    /// Level 1 runs at full resolution and both blocks expand channels 6× (`dim=48` → 288), and
    /// `patchify` copies that again, so a single 1080p frame peaks around **40 GB** of MLX
    /// allocation. Measured, and predicted to within ~10% by arithmetic — see PORT-STATUS.md.
    ///
    /// Crop-valid tiling (no feathered blend): each tile is run with `overlap` pixels of context on
    /// every side and only its **core** is written out. Given `overlap ≥ receptive field` this is
    /// *exactly* equal to the full-frame result, not an approximation — no seams to blend away.
    /// That holds here because FSAS is per-8×8-patch, not global: cross-tile influence travels only
    /// through 3×3 convs and two downsamples. The `--tile` gate measures the overlap needed.
    ///
    /// Each tile is evaluated and released before the next starts (MLX otherwise accumulates
    /// unbounded residency across a long sequential graph — the documented tile-loop failure).
    ///
    /// - Parameters:
    ///   - tile: core+context extent per tile. Multiples of 32 avoid extra internal padding.
    ///   - overlap: context pixels on each side, discarded from the output.
    /// Defaults chosen by measurement at 1080p (`--bench`), MLX peak / process phys:
    /// `128 → 3.58/3.51 GB` (~510 tiles) · `192 → 3.84/4.32 GB` (~240) · **`256 → 4.47/8.07 GB`
    /// (~60)** · `384 → 5.82/10.18 GB` (~30). 256 is the knee: it fits a 16 GB Mac with room while
    /// keeping the tile count — and therefore wall-clock — reasonable. A 512 tile costs 8.7 GB MLX /
    /// 33 GB phys **per tile** and exhausts even a 137 GB machine across a 1080p frame (SIGTRAP).
    ///
    /// `tile` is a genuine memory lever with little quality cost, so it is the natural knob for a
    /// `BudgetAware` tier to turn down under pressure.
    /// - Parameter onTile: invoked once per tile as `(completed, total)` **before** that tile runs.
    ///   The tile loop is a genuine iterative seam, so this is where a wrapper hangs cooperative
    ///   cancellation and per-tile progress. Throwing from it aborts the run — the throw propagates
    ///   unchanged, which is what lets the engine distinguish a user cancel from a package error.
    ///   Kept as a closure rather than making the core `async`/engine-aware: `FFTformerMLXCore`
    ///   depends on MLX alone and stays usable standalone.
    public func restoreTiled(_ image: MLXArray, tile: Int = 256, overlap: Int = 32,
                             onTile: ((Int, Int) throws -> Void)? = nil) rethrows -> MLXArray {
        // 🔑 TILE GEOMETRY MUST BE 32-ALIGNED. Measured, and the effect is large: a sweep at
        // tile=256 gave ~26 dB at overlaps 0/32/64/96 but ~20.6 dB at 16/48/112 — the error tracks
        // `overlap % 32` exactly, with no receptive-field trend at all.
        //
        // Why: every FFT block decomposes its input into 8×8 patches measured from the TILE origin,
        // and level 3 runs at 1/4 resolution, so one patch there spans 32 full-res pixels. If a
        // tile's origin is not ≡ 0 (mod 32) the whole patch grid shifts phase against the
        // full-frame decomposition, and the FFT sees genuinely different windows — a discretization
        // mismatch, not an approximation that more context would fix.
        //
        // The origin is `coreY - overlap`, so BOTH the step and the overlap must be multiples of 32.
        let tile = max(64, (tile / 32) * 32)
        let overlap = (overlap / 32) * 32
        precondition(tile > 2 * overlap, "tile (\(tile)) must exceed 2·overlap (\(2 * overlap))")

        let (b, h, w, c) = (image.dim(0), image.dim(1), image.dim(2), image.dim(3))

        // Small enough to do in one pass — identical result, no tiling bookkeeping.
        if h <= tile && w <= tile { return restore(image) }

        let step = tile - 2 * overlap   // a multiple of 32 whenever tile and overlap are

        // Feathered accumulation rather than crop-valid hard writes.
        //
        // Crop-valid would be exact only if `overlap ≥ receptive field`, and this model's RF is
        // large — 3×3 convs through 6+6+12 encoder and 12+6+6+4 decoder/refinement blocks, several
        // of them at 1/4 resolution, put it in the low hundreds of pixels. An overlap that big
        // leaves almost no core. Feathering instead blends the overlapping estimates, which removes
        // seams without pretending to reproduce a full-frame result that is itself unattainable at
        // production sizes.
        var acc = MLXArray.zeros([b, h, w, c], dtype: .float32)
        var wsum = MLXArray.zeros([1, h, w, 1], dtype: .float32)

        let rows = (h + step - 1) / step
        let cols = (w + step - 1) / step
        let total = rows * cols
        var done = 0

        for coreY in stride(from: 0, to: h, by: step) {
            let inY0 = max(0, coreY - overlap)
            let inY1 = min(h, coreY + step + overlap)

            for coreX in stride(from: 0, to: w, by: step) {
                try onTile?(done, total)
                done += 1

                let inX0 = max(0, coreX - overlap)
                let inX1 = min(w, coreX + step + overlap)

                let restored = restore(image[0..., inY0 ..< inY1, inX0 ..< inX1, 0...])
                let weight = Self.featherWeights(height: inY1 - inY0, width: inX1 - inX0,
                                                 ramp: overlap,
                                                 topEdge: inY0 == 0, bottomEdge: inY1 == h,
                                                 leftEdge: inX0 == 0, rightEdge: inX1 == w)

                acc[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] =
                    acc[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] + restored.asType(.float32) * weight
                wsum[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] =
                    wsum[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] + weight

                // Realize and release before the next tile, so peak stays one-tile-sized. Without
                // this MLX accumulates unbounded residency across a long sequential graph.
                eval(acc, wsum)
                MLX.Memory.clearCache()
            }
        }
        return clip(acc / maximum(wsum, MLXArray(1e-8)), min: 0, max: 1).asType(image.dtype)
    }

    /// Separable linear ramp: 1 across the interior, falling to ~0 over `ramp` pixels at each inner
    /// edge. Image-boundary edges are NOT ramped — nothing overlaps them, so a falloff there would
    /// divide by a small weight and amplify noise at the frame border.
    private static func featherWeights(height: Int, width: Int, ramp: Int,
                                       topEdge: Bool, bottomEdge: Bool,
                                       leftEdge: Bool, rightEdge: Bool) -> MLXArray {
        func profile(_ n: Int, _ startFlat: Bool, _ endFlat: Bool) -> [Float] {
            var v = [Float](repeating: 1, count: n)
            guard ramp > 0 else { return v }
            let r = min(ramp, n / 2)
            for i in 0 ..< r {
                let t = (Float(i) + 0.5) / Float(r)
                if !startFlat { v[i] = t }
                if !endFlat { v[n - 1 - i] = min(v[n - 1 - i], t) }
            }
            return v
        }
        let vy = profile(height, topEdge, bottomEdge)
        let vx = profile(width, leftEdge, rightEdge)
        let y = MLXArray(vy, [1, height, 1, 1])
        let x = MLXArray(vx, [1, 1, width, 1])
        return y * x
    }

    /// Loads converted safetensors weights under the strict verifier.
    ///
    /// `.all` rather than `.noUnusedKeys`: the module tree is an exact 535-tensor match for the
    /// released checkpoint (gate S0), so anything less strict would let a silent structural drift
    /// through. Weights are expected in the converted MLX layout — conv `(O,kH,kW,I)`, FFT masks
    /// `(8,5,C)`; see `oracle/convert.py`.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
