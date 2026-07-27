//
//  main.swift
//  mlx-fftformer-swift / FFTformerValidate
//
//  Drives the package through the **real `MLXServeEngine`** and reports the authoritative split
//  footprint, so the manifest can stop carrying an estimate.
//
//  Why this exists rather than the gate's `--bench`: `--bench` calls the core model directly and
//  reads MLX-pool memory. Two problems. It bypasses register/prepare and the governor, and MLX-pool
//  peak **under-reads the admission basis by ~2.7×** — the BiRefNet re-baseline that forced every
//  optimizer-family manifest to be re-measured (BiRefNet best went 18.3 → 47.8 GB when measured
//  properly). The number that matters is process `phys_footprint`.
//
//  This target uses `MLXEngineTestKit.ValidationHarness`, which is the same code path and the same
//  metric the (now archived) MLXEngineImage validation app used: 150 ms `phys_footprint` sampling,
//  the resident floor read **post-load / pre-run**, and the `[label] SPLIT floor= peak= act=` line
//  the manifests were re-baselined from. `package-efficiency.md` explicitly blesses an
//  xcodebuild-or-`swift run` executable reading `phys_footprint` as equivalent to the in-app number.
//
//  ⚠️ One honest caveat: a CLI process carries no AppKit/Metal-view overhead, so absolute `floor`
//  and `peak` sit a few hundred MB below a GUI app's. That is conservative in the WRONG direction
//  for admission, so the manifest adds margin. `activation = peak − floor` — the dominant term for
//  a full-frame restorer — is process-shape independent.
//
//  Usage:  swift run fftformer-validate <weights.safetensors> [image.png] [width] [height]
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXServeCore
import MLXEngineTestKit
import MLXFFTformer

setvbuf(stdout, nil, _IONBF, 0)

let args = Array(CommandLine.arguments.dropFirst())
guard let weightsPath = args.first else {
    print("usage: fftformer-validate <weights.safetensors> [image.png] [width] [height]")
    exit(2)
}
let imagePath = args.count > 1 ? args[1] : nil
let width = args.count > 2 ? Int(args[2]) ?? 1920 : 1920
let height = args.count > 3 ? Int(args[3]) ?? 1080 : 1080

/// A real PNG if one was supplied, else a synthetic structured image at the requested size.
/// Synthetic is fine for a MEMORY measurement — footprint depends on dimensions, not content — and
/// it keeps the envelope explicit rather than whatever a fixture happens to be.
func makeImage() -> MLXToolKit.Image {
    if let imagePath, let data = FileManager.default.contents(atPath: imagePath) {
        let src = CGImageSourceCreateWithData(data as CFData, nil)
        let cg = src.flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        print("input: \(imagePath) (\(cg?.width ?? 0)x\(cg?.height ?? 0))")
        return MLXToolKit.Image(format: .png, data: data, width: cg?.width, height: cg?.height)
    }
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0 ..< height {
        for x in 0 ..< width {
            let i = (y * width + x) * 4
            let fx = Float(x) / Float(width), fy = Float(y) / Float(height)
            bytes[i + 0] = UInt8(max(0, min(255, (0.5 + 0.3 * sin(fx * 11)) * 255)))       // B
            bytes[i + 1] = UInt8(max(0, min(255, (0.5 + 0.3 * cos(fy * 9)) * 255)))        // G
            bytes[i + 2] = UInt8(max(0, min(255, (0.5 + 0.3 * sin((fx + fy) * 7)) * 255))) // R
            bytes[i + 3] = 255
        }
    }
    print("input: synthetic \(width)x\(height)")
    return MLXToolKit.Image.rawBGRA8(data: Data(bytes), width: width, height: height)
}

@MainActor
func main() async {
    // Match the shipping configuration: Forge constructs the engine `.blocking`, so a package that
    // would be refused in production must be refused here too.
    let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)

    let config = FFTformerConfiguration(weightsURL: URL(fileURLWithPath: weightsPath))
    let request = ImageRestoreRequest(image: makeImage())

    do {
        let result = try await ValidationHarness.run(
            engine: engine,
            registration: FFTformerRestorePackage.registration,
            configuration: config,
            capability: .imageRestore,
            request: request,
            isolate: true,
            clearCache: { MLX.GPU.clearCache() },
            inputSummary: "\(width)x\(height)",
            heartbeatLabel: "fftformer")

        print("")
        print(result.run.splitLogLine("fftformer-\(config.variant.rawValue)"))
        print("")
        print("  DECLARE  residentBytes        = \(result.run.residentFloorBytes)")
        print("           peakActivationBytes  = \(result.run.activationBytes)")
        if result.run.retainedAfterRunBytes > 200_000_000 {
            print("  ⚠️ retains \(result.run.retainedAfterRunBytes) B after run+clearCache — "
                + "a live model holding intermediates. Belongs in the transient, not residency.")
        }
        if let out = result.response as? ImageRestoreResponse {
            print("  output: \(out.image.data.count) bytes, \(out.image.width ?? 0)x\(out.image.height ?? 0)")
        }
    } catch {
        print("❌ validation failed: \(error)")
        exit(1)
    }
}

await main()
