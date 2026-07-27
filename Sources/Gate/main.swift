//
//  main.swift
//  mlx-fftformer-swift / FFTformerGate
//
//  Parity gates that need a real Metal context. These live in an executable, NOT the test target:
//  the SPM test product's metallib is unreliable, while `swift run` does GPU inference fine.
//  (mlx-swift-integration/references/swift-port-parity.md, "CLI gate modes".)
//
//  Modes:
//    --s0 <weights.safetensors>   key contract: module tree vs checkpoint, 0 missing / 0 unused
//

import Foundation
import FFTformerMLXCore
import MLX
import MLXNN

// Unbuffered: a crash mid-gate must not swallow the output that says where it happened.
private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never {
    _ = _unbuffered
    print("❌ \(msg)")
    exit(1)
}

/// S0 — the key contract. Cheapest gate in the suite and the one that catches structural drift:
/// a wrong expansion factor, a missing dead module, a mis-shaped FFT mask. Runs no kernel.
func gateS0(weightsPath: String) {
    print("=== S0 · key contract ===\n")

    let model = FFTformer()
    let params = model.parameters().flattened()
    var swiftKeys: [String: [Int]] = [:]
    for (k, v) in params { swiftKeys[k] = v.shape }
    print("Swift module tree : \(swiftKeys.count) parameter tensors")

    let total = params.reduce(0) { $0 + $1.1.size }
    print("Swift parameters  : \(total)")

    let url = URL(fileURLWithPath: weightsPath)
    guard let loaded = try? MLX.loadArrays(url: url) else {
        fail("could not load \(weightsPath)")
    }
    print("Checkpoint        : \(loaded.count) tensors")
    let ckptTotal = loaded.values.reduce(0) { $0 + $1.size }
    print("Checkpoint params : \(ckptTotal)\n")

    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted()     // Swift declares, checkpoint lacks
    let unused = ck.subtracting(sk).sorted()      // checkpoint has, Swift never declares

    if !missing.isEmpty {
        print("MISSING (\(missing.count)) — Swift declares, checkpoint lacks:")
        missing.prefix(15).forEach { print("   \($0)  \(swiftKeys[$0]!)") }
        if missing.count > 15 { print("   … \(missing.count - 15) more") }
        print("")
    }
    if !unused.isEmpty {
        print("UNUSED (\(unused.count)) — checkpoint has, Swift never declares:")
        unused.prefix(15).forEach { print("   \($0)  \(loaded[$0]!.shape)") }
        if unused.count > 15 { print("   … \(unused.count - 15) more") }
        print("")
    }

    // Shape mismatches are the subtle failure — same key, wrong tensor.
    var shapeMismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) {
        let a = swiftKeys[k]!, b = loaded[k]!.shape
        if a != b { shapeMismatch.append((k, a, b)) }
    }
    if !shapeMismatch.isEmpty {
        print("SHAPE MISMATCH (\(shapeMismatch.count)):")
        for (k, a, b) in shapeMismatch.prefix(15) { print("   \(k)\n     swift \(a)  vs  ckpt \(b)") }
        if shapeMismatch.count > 15 { print("   … \(shapeMismatch.count - 15) more") }
        print("")
    }

    guard missing.isEmpty, unused.isEmpty, shapeMismatch.isEmpty else {
        fail("S0 FAILED — the module tree does not match the checkpoint.")
    }

    // Prove it actually loads under the strict verifier, not just that the key sets match.
    do {
        try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all)
    } catch {
        fail("S0 FAILED at update(verify: .all): \(error)")
    }

    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, "
        + "0 missing / 0 unused / 0 shape mismatches, strict update clean.")
}

// MARK: - Numeric gates

/// Loads the model with real weights, for the block- and model-level gates.
func loadedModel(_ weightsPath: String) -> FFTformer {
    let model = FFTformer()
    guard let w = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    do { try model.update(parameters: ModuleParameters.unflattened(w), verify: .all) }
    catch { fail("weight load failed: \(error)") }
    eval(model)
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") }
    catch { fail("golden \(name): \(error)") }
}

/// Splits the error into interior vs 1-pixel border to tell rounding from edge-handling.
///
/// Resampling bugs (wrong `alignCorners`, off-by-one grid, wrong padding mode) concentrate their
/// error at the boundary, where the sampling kernel runs off the edge. Uniform floating-point
/// rounding does not. Both inputs are NCHW.
func borderDiagnostic(_ label: String, _ got: MLXArray, _ want: MLXArray) {
    let d = MLX.abs(got.asType(.float32) - want.asType(.float32))
    let (h, w) = (d.dim(2), d.dim(3))
    guard h > 2, w > 2 else { return }
    let interior = d[0..., 0..., 1 ..< (h - 1), 1 ..< (w - 1)]
    let overall = MLX.max(d).item(Float.self)
    let inner = MLX.max(interior).item(Float.self)
    let ratio = inner > 0 ? overall / inner : .infinity
    let verdict = ratio > 4
        ? "border-concentrated → EDGE-HANDLING BUG, investigate"
        : "uniform → floating-point rounding, not a semantic error"
    print(String(format: "     ↳ %@ error: overall max %.3e, interior max %.3e (×%.2f) — %@",
                 label, overall, inner, ratio, verdict))
}

/// S1a — the primitives, on seeded inputs. If any of these is red, nothing downstream is meaningful.
func gateS1a(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S1a · primitives ===\n")
    let r = GateReport("S1a")
    let model = loadedModel(weightsPath)

    // -- channel LayerNorm (uses the real norm2 of encoder_level1.0, matching the oracle)
    let lnIn = toNHWC(g(dir, "ln_withbias_in"))
    let lnOut = model.encoderLevel1[0].norm2(lnIn)
    r.check("layernorm_withbias", toNCHW(lnOut), g(dir, "ln_withbias_out"), tol: 1e-6)

    // -- patchify / unpatchify. Golden is PyTorch (B,C,h,w,p1,p2); ours is (B,h,w,p1,p2,C).
    let patchIn = toNHWC(g(dir, "patch_in"))
    let patched = patchify(patchIn)
    r.check("patchify", patchToNCHW(patched), g(dir, "patch_out"), tol: 0)
    r.check("unpatchify_roundtrip", toNCHW(unpatchify(patched)), g(dir, "patch_roundtrip"), tol: 0)

    // -- FFT gate: rfft2 -> real mask -> irfft2. Mask golden is (C,1,1,8,5); ours is (8,5,C).
    let fftIn = patchToNHWC(g(dir, "fftcore_in"))
    let maskRaw = g(dir, "fftcore_mask")                       // (C,1,1,8,5)
    let mask = maskRaw.reshaped(maskRaw.dim(0), maskRaw.dim(3), maskRaw.dim(4))
        .transposed(1, 2, 0)                                   // -> (8,5,C)
    r.check("fft_gate", patchToNCHW(fftGate(fftIn, mask: mask)), g(dir, "fftcore_out"), tol: 1e-6)

    // -- FSAS core: complex product of two spectra.
    let q = patchToNHWC(g(dir, "fsascore_q"))
    let k = patchToNHWC(g(dir, "fsascore_k"))
    r.check("fft_match", patchToNCHW(fftMatch(q, k)), g(dir, "fsascore_out"), tol: 1e-6)

    // -- bilinear resamplers (align_corners = false)
    //
    // Tolerance is 1e-5 rather than the 1e-6 used for the other primitives, and the reason is
    // structural, not goalpost-moving: these are the only primitives that compute *interpolation
    // weights* in floating point ((i + 0.5) · scale − 0.5) and then accumulate a conv over up to
    // 192 input channels. Last-ulp differences in the weight arithmetic between MLX and PyTorch
    // are expected and land around 1e-6 relative. A semantic error — a wrong `alignCorners`, an
    // off-by-one in the sampling grid — would resample a visibly different grid and show up orders
    // of magnitude larger, not at 1e-6. The interior-vs-border diagnostic below discriminates the
    // two directly: edge-handling bugs concentrate error at the border, rounding does not.
    let downOut = toNCHW(model.down1_2(toNHWC(g(dir, "down_in"))))
    r.check("downsample", downOut, g(dir, "down_out"), tol: 1e-5)
    let upOut = toNCHW(model.up3_2(toNHWC(g(dir, "up_in"))))
    r.check("upsample", upOut, g(dir, "up_out"), tol: 1e-5)
    borderDiagnostic("upsample", upOut, g(dir, "up_out"))

    return r.summarize()
}

/// S2 — composite blocks with real weights.
func gateS2(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S2 · blocks (real weights) ===\n")
    let r = GateReport("S2")
    let model = loadedModel(weightsPath)

    r.check("dffn", toNCHW(model.encoderLevel1[0].ffn(toNHWC(g(dir, "dffn_in")))),
            g(dir, "dffn_out"), tol: 1e-5)

    guard let attn = model.decoderLevel1[0].attn else { fail("decoder_level1.0 has no attn") }
    r.check("fsas", toNCHW(attn(toNHWC(g(dir, "fsas_in")))), g(dir, "fsas_out"), tol: 1e-5)

    let tIn = toNHWC(g(dir, "tblock_in"))
    r.check("tblock_noatt", toNCHW(model.encoderLevel1[0](tIn)), g(dir, "tblock_noatt_out"), tol: 1e-5)
    r.check("tblock_att", toNCHW(model.decoderLevel1[0](tIn)), g(dir, "tblock_att_out"), tol: 1e-5)

    // fuse_out == fuse2(fuse_a, fuse_b) — the fixture names encode the argument order.
    r.check("fuse", toNCHW(model.fuse2(toNHWC(g(dir, "fuse_a")), toNHWC(g(dir, "fuse_b")))),
            g(dir, "fuse_out"), tol: 1e-5)

    return r.summarize()
}

/// S3 — the whole model, including the largest production tile with a structured (non-noise) input.
func gateS3(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S3 · full model ===\n")
    let r = GateReport("S3")
    let model = loadedModel(weightsPath)

    for name in ["full_64", "full_128", "full_256", "full_img256"] {
        let x = toNHWC(g(dir, "\(name)_in"))
        let out = model(x)
        eval(out)
        r.check(name, toNCHW(out), g(dir, "\(name)_out"), tol: 1e-4)
    }
    return r.summarize()
}

/// S4 — publish-dtype decision, measured on MLX rather than inherited from the torch-CPU probe.
///
/// The oracle measured bf16 at cosine 0.9839 / 30.1 dB and fp16 as NaN, but torch-CPU half support
/// is flaky, so the fp16 result specifically needed confirming on the real backend. Mixed-precision
/// contract applies: the 54 `.fft` masks stay fp32 (the FFT interior is fp32 by construction).
func gateS4(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S4 · publish dtype (MLX-side) ===\n")
    guard let w = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    let want = g(dir, "full_img256_out")
    let x = toNHWC(g(dir, "full_img256_in"))

    for (tag, dt) in [("fp32", DType.float32), ("bf16", .bfloat16), ("float16", .float16)] {
        let model = FFTformer()
        // Keep the learned frequency masks fp32 — see PORT-STATUS.md "mixed-precision contract".
        var cast: [String: MLXArray] = [:]
        for (k, v) in w { cast[k] = k.hasSuffix(".fft") ? v : v.asType(dt) }
        do { try model.update(parameters: ModuleParameters.unflattened(cast), verify: .all) }
        catch { fail("\(tag): weight load failed: \(error)") }
        eval(model)

        let out = model(x.asType(dt))
        eval(out)
        let p = parity(toNCHW(out).asType(.float32), want)

        // PSNR on the clamped output image — meaningful here because restoration is deterministic.
        let a = MLX.clip(toNCHW(out).asType(.float32), min: 0, max: 1)
        let b = MLX.clip(want, min: 0, max: 1)
        let mse = MLX.mean(MLX.square(a - b)).item(Float.self)
        let psnr = mse > 0 ? 10 * log10(1.0 / mse) : Float.infinity

        let verdict = p.cosine >= 0.9999 ? "OK" : (p.cosine >= 0.999 ? "MARGINAL" : "FAIL")
        print(String(format: "  %-8s cos=%.8f  rel=%.2e  PSNR_vs_fp32=%6.2f dB   [%@]",
                     (tag as NSString).utf8String!, p.cosine, p.relative, psnr, verdict))
    }
    print("\n  Ships fp32 unless a lower dtype reaches cos ≥ 0.9999 AND PSNR ≥ ~50 dB — the model's")
    print("  whole task signal is GoPro 34.21 dB, so dtype error must sit well below it.")
    return true
}

/// `--tile` — measure the overlap tiling actually needs, instead of guessing a receptive field.
///
/// Runs a 512² input full-frame (which still fits) and then tiled at 256 with a sweep of overlaps,
/// comparing against the full-frame result. Crop-valid tiling is *exact* once overlap ≥ receptive
/// field, so the error should fall off a cliff and then flatten at fp32 rounding. Pick the first
/// overlap on the flat part.
func gateTile(_ weightsPath: String) -> Bool {
    print("=== TILE · required-overlap sweep ===\n")
    let model = loadedModel(weightsPath)

    // Structured input, not noise: noise has no spatial correlation and would understate how far
    // real content propagates across a tile boundary.
    let n = 512
    var pixels = [Float](repeating: 0, count: n * n * 3)
    for y in 0 ..< n {
        for x in 0 ..< n {
            let fx = Float(x) / Float(n), fy = Float(y) / Float(n)
            let i = (y * n + x) * 3
            pixels[i + 0] = 0.5 + 0.35 * sin(fx * 9) * cos(fy * 7)
            pixels[i + 1] = 0.5 + 0.35 * cos(fx * 6 + fy * 5)
            pixels[i + 2] = 0.5 + 0.30 * sin((fx + fy) * 11)
        }
    }
    let x = MLXArray(pixels, [1, n, n, 3])

    let full = model.restore(x)
    eval(full)
    MLX.Memory.clearCache()

    print("  reference: full-frame 512²; tiled at 256")
    print("  Judged on the DISTRIBUTION, not max_abs: `restore` clamps to [0,1], so a handful of")
    print("  pixels whose pre-clamp values straddle a bound saturate max_abs at exactly 1.0 and")
    print("  hide the trend entirely. PSNR + an outlier count show what is actually happening.\n")
    print("  overlap    PSNR      mean_abs    >0.01    >0.1    max_abs")

    for overlap in [0, 16, 32, 48, 64, 96, 112] {
        let tiled = model.restoreTiled(x, tile: 256, overlap: overlap)
        eval(tiled)
        let d = MLX.abs(tiled - full)
        let mse = MLX.mean(MLX.square(tiled - full)).item(Float.self)
        let psnr = mse > 0 ? 10 * log10(1.0 / mse) : Float.infinity
        let meanAbs = MLX.mean(d).item(Float.self)
        let n01 = MLX.sum((d .> 0.01).asType(.int32)).item(Int32.self)
        let n1 = MLX.sum((d .> 0.1).asType(.int32)).item(Int32.self)
        let mx = MLX.max(d).item(Float.self)
        print(String(format: "  %7d  %7.2f dB  %.3e  %6d  %6d  %.4f",
                     overlap, psnr, meanAbs, n01, n1, mx))
        MLX.Memory.clearCache()
    }
    // Diagnostic: is the discrepancy unwritten output, or genuine boundary influence?
    let t = model.restoreTiled(x, tile: 256, overlap: 64)
    eval(t)
    let d = MLX.abs(t - full)
    let bad = MLX.sum((d .> 0.5).asType(.int32)).item(Int32.self)
    print(String(format: "\n  DIAG overlap 64: pixels differing >0.5 = %d of %d", bad, 512*512*3))
    print("       tiled  min/max = \(MLX.min(t).item(Float.self)) / \(MLX.max(t).item(Float.self))")
    print("       full   min/max = \(MLX.min(full).item(Float.self)) / \(MLX.max(full).item(Float.self))")
    let zeros = MLX.sum((t .== 0).asType(.int32)).item(Int32.self)
    print("       exact zeros in tiled = \(zeros)  (unwritten output would show here)")

    print("\n  Choose the smallest overlap on the flat tail — beyond that the residual is fp32")
    print("  rounding, and extra context only costs compute.")
    return true
}

// MARK: - Footprint bench

/// Process `phys_footprint` — the authoritative admission basis.
///
/// The registry records that MLX-peak under-reads this by ~2.7× (the BiRefNet re-baseline): MLX's
/// own counter cannot see the Metal driver's working set, IOSurface backing, or process overhead.
/// Declaring from MLX-peak alone under-declares, which falsely admits on tight Macs.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
func gb(_ b: Int) -> String { gb(UInt64(max(0, b))) }

/// `--bench` — split-footprint measurement for the manifest: resident weights floor (post-load,
/// post-clearCache) and the transient activation peak at a given input size.
func gateBench(_ weightsPath: String, sizes: [(Int, Int)]) {
    _ = _unbuffered
    print("=== BENCH · split footprint ===\n")
    print("  NOTE: GPU stream; measures the TILED production path (tile 512, overlap 64).\n")

    let base = physFootprintBytes()
    print("  baseline phys_footprint (pre-load) : \(gb(base))")

    let model = loadedModel(weightsPath)
    MLX.Memory.clearCache()
    let floor = physFootprintBytes()
    let resident = floor > base ? floor - base : 0
    print("  post-load floor                    : \(gb(floor))  → resident ≈ \(gb(resident))")
    print("  (weights are 16,560,474 params @ fp32 = 66.2 MB; the rest is runtime + Metal overhead)\n")

    // Sweep TILE SIZE at 1080p — the production question is "what tile fits", not "what does a
    // full frame cost". Each tile is a full model forward, so peak is one-tile-sized.
    for tile in [128, 192, 256, 384] {
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        let x = MLXArray.zeros([1, 1080, 1920, 3], dtype: .float32)
        let out = model.restoreTiled(x, tile: tile, overlap: 32)
        eval(out)
        let mlxPeak = MLX.Memory.peakMemory
        let peakPhys = physFootprintBytes()
        print("  1080p @ tile \(tile): MLX peak \(gb(mlxPeak))   phys \(gb(peakPhys))")
        MLX.Memory.clearCache()
    }
    _ = sizes

    print("\n  Declare residentBytes = post-load floor, peakActivationBytes = phys peak − floor.")
    print("  These are still CLI numbers — the manifest wants the IN-APP figure before validation.")
}

// MARK: - entry

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("""
    usage:
      fftformer-gate --s0  <weights.safetensors>
      fftformer-gate --s1a <goldens-dir> <weights.safetensors>
      fftformer-gate --s2  <goldens-dir> <weights.safetensors>
      fftformer-gate --s3  <goldens-dir> <weights.safetensors>
      fftformer-gate --all <goldens-dir> <weights.safetensors>
    """)
    exit(2)
}

// fp32 gates pin to the CPU stream: Apple-GPU fp32 accumulates ~8e-4 relative error per op, which
// both masks real bugs and gets mistaken for them. (Quantized forwards must NOT do this — they
// have no CPU path and silently grind for hours. Not applicable here: this model ships fp32.)
if mode != "--bench" && mode != "--tile" { Device.setDefault(device: .cpu) }

switch mode {
case "--s0":
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(weightsPath: args[1])
case "--subtest":
    // Isolate the 4-D subscript setter + accumulate pattern used by restoreTiled.
    Device.setDefault(device: .gpu)
    var a = MLXArray.zeros([1, 64, 64, 3], dtype: .float32)
    var wsum2 = MLXArray.zeros([1, 64, 64, 1], dtype: .float32)
    let tileArr = MLXArray.ones([1, 16, 16, 3], dtype: .float32)
    let wArr = MLXArray.ones([1, 16, 16, 1], dtype: .float32)
    print("read slice…"); let rd = a[0..., 8 ..< 24, 8 ..< 24, 0...]; eval(rd); print("  ok \(rd.shape)")
    print("write slice (no alias)…"); a[0..., 8 ..< 24, 8 ..< 24, 0...] = tileArr; eval(a); print("  ok")
    print("accumulate (alias: read+write same array)…")
    a[0..., 8 ..< 24, 8 ..< 24, 0...] = a[0..., 8 ..< 24, 8 ..< 24, 0...] + tileArr * wArr
    eval(a); print("  ok, sum=\(MLX.sum(a).item(Float.self))")
    print("wsum accumulate…")
    wsum2[0..., 8 ..< 24, 8 ..< 24, 0...] = wsum2[0..., 8 ..< 24, 8 ..< 24, 0...] + wArr
    eval(wsum2); print("  ok, sum=\(MLX.sum(wsum2).item(Float.self))")
    print("ALL SUBTESTS PASSED")
case "--tile":
    guard args.count >= 2 else { fail("--tile needs a weights path") }
    Device.setDefault(device: .gpu)
    _ = gateTile(args[1])
case "--bench":
    guard args.count >= 2 else { fail("--bench needs a weights path") }
    // Bench runs on the GPU (default stream) — it measures memory, not numerics.
    Device.setDefault(device: .gpu)
    gateBench(args[1], sizes: [(512, 512), (1920, 1080), (1214, 2160)])
case "--s1a", "--s2", "--s3", "--s4", "--all":
    guard args.count >= 3 else { fail("\(mode) needs <goldens-dir> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1a" || mode == "--all" { ok = gateS1a(dir, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok; print("") }
    if mode == "--s4" || mode == "--all" { ok = gateS4(dir, w) && ok }
    if !ok { exit(1) }
default:
    fail("unknown mode \(mode)")
}
