// GPU-lane gate for mlx's lossy Winograd conv2d window (Sources/FFTformerMLXCore/
// WinogradFreeConv2d.swift): one production 256² tile, fp32, on the CPU stream (exact-class
// reference) and on the GPU with the conv route on (default) and off (raw). The tile is the
// bottom-left one of a 1024×768 centre crop of DIV2K 0801 with a 9-tap horizontal box blur (8-bit),
// where the tiled run measured a border hotspot: raw GPU 6 of 255 levels vs 1 with
// MLX_ENABLE_TF32=0 (2026-09-24, M5 Max, mlx-swift 0.31.6). fftformer-gate pins the CPU stream.
//
// Run: FFT_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests
// Overrides: FFT_WEIGHTS (model.safetensors), FFT_REAL_IMAGE.

import CoreGraphics
import FFTformerMLXCore
import Foundation
import ImageIO
import MLX
import XCTest

final class GPULaneTests: XCTestCase {
    func testBorderTileGPUvsCPU() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["FFT_LANE"] == "1", "set FFT_LANE=1 to run")
        let weights = env["FFT_WEIGHTS"]
            ?? "/Volumes/Satechi/Models/models/mlx-community/FFTformer-GoPro-fp32/model.safetensors"
        let image = URL(fileURLWithPath: env["FFT_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")
        let model = FFTformer()
        try model.loadWeights(from: URL(fileURLWithPath: weights))

        guard let src = CGImageSourceCreateWithURL(image as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "FFT", code: 1) }
        let (iw, ih, cw, ch) = (cg.width, cg.height, 1024, 768)
        var rgba = [UInt8](repeating: 0, count: iw * ih * 4)
        let ctx = CGContext(
            data: &rgba, width: iw, height: ih, bitsPerComponent: 8, bytesPerRow: iw * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: iw, height: ih))
        let (x0, y0) = ((iw - cw) / 2, (ih - ch) / 2)
        // 1024×768 crop, 9-tap horizontal box blur with edge clamp, in 8-bit (as the probe did).
        var crop = [UInt8](repeating: 0, count: cw * ch * 3)
        for y in 0..<ch {
            for x in 0..<cw {
                for c in 0..<3 {
                    var acc = 0
                    for k in -4...4 {
                        let xx = min(cw - 1, max(0, x + k))
                        acc += Int(rgba[((y0 + y) * iw + (x0 + xx)) * 4 + c])
                    }
                    crop[(y * cw + x) * 3 + c] = UInt8((Double(acc) / 9).rounded())
                }
            }
        }
        // The bottom-left 256² production tile.
        let (tx, ty, t) = (0, ch - 256, 256)
        var tile = [Float](repeating: 0, count: t * t * 3)
        for y in 0..<t {
            for x in 0..<t {
                for c in 0..<3 { tile[(y * t + x) * 3 + c] = Float(crop[((ty + y) * cw + tx + x) * 3 + c]) / 255 }
            }
        }
        let xIn = MLXArray(tile, [1, t, t, 3])

        let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = clip(model(xIn), min: 0, max: 1)
            eval(r)
            return r
        }
        Memory.clearCache()
        func run(_ route: FFTformerConvRoute) -> (MLXArray, Double) {
            model.convRoute = route
            var y = clip(model(xIn), min: 0, max: 1)
            eval(y)
            let t0 = Date()
            for _ in 0..<3 { y = clip(model(xIn), min: 0, max: 1); eval(y) }
            return (y, Date().timeIntervalSince(t0) / 3 * 1000)
        }
        let (r, tr) = run(.conv3d), (w, tw) = run(.winograd)
        model.convRoute = .conv3d
        func stats(_ a: MLXArray) -> (rel: Float, levels: Int, text: String) {
            let d = a - ref
            let rel = sqrt(sum(d * d)) / sqrt(sum(ref * ref))
            let lv = abs(floor(a * 255) - floor(ref * 255)).max()   // production truncation
            eval(rel, lv)
            let (rv, lvv) = (rel.item(Float.self), Int(lv.item(Float.self)))
            return (rv, lvv, String(format: "relL2 %.2e  8-bit max %d levels", rv, lvv))
        }
        let sR = stats(r), sW = stats(w)
        print("[FFTformer border tile 256² of a 1024×768 motion-blurred crop, fp32, GPU vs CPU lane]")
        print(String(format: "  conv3d route   %@  %6.1f ms", sR.text, tr))
        print(String(format: "  raw Winograd   %@  %6.1f ms", sW.text, tw))
        XCTAssertLessThan(sR.rel, 1e-5, "conv3d route vs CPU lane")
        XCTAssertLessThanOrEqual(sR.levels, 1, "conv3d route: at most 1 level (truncation) off")
    }
}
