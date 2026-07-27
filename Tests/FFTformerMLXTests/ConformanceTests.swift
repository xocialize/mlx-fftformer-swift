// ConformanceTests.swift — FFTformer through the engine's offline gates (no MLX kernels run).
//
//   MAT-1..5  the WeightSourcing declarations (fresh-machine materialization)
//   CAN-1..3  cooperative cancellation: pre-cancelled run() + the checkpoint-cadence declaration
//   C-manifest  license, footprint and surface sanity
//
// CAN-3 note: unlike NAFNet on the same capability, FFTformer's `run()` has a GENUINE iterative
// seam — the tile loop. Full-frame is not viable (≈40 GB at 1080p), so `restoreTiled` is the
// production path and it checkpoints once per tile via the core's `onTile` hook. The declared
// cadence names that real loop rather than only the phase boundaries.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXFFTformer

final class ConformanceTests: XCTestCase {

    // MARK: - MAT — first-run materialization declarations

    func testMATGate() {
        // Fresh: no store root, no explicit weights → everything must read as missing.
        let fresh = FFTformerConfiguration()
        let report = MaterializationConformance.check(freshConfiguration: fresh)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclaredForEveryVariant() {
        for variant in FFTformerVariant.allCases {
            let cfg = FFTformerConfiguration(variant: variant)
            let sources = cfg.weightSources
            XCTAssertEqual(sources.count, 1, "\(variant): expected exactly one weight source")
            XCTAssertEqual(sources[0].repo, variant.repo)
            // Globs are declared so a half-materialized source reads as missing rather than
            // silently degrading (MS-2).
            XCTAssertEqual(sources[0].matching, ["model.safetensors"], "\(variant)")
        }
    }

    /// The `weightsURL` escape hatch must short-circuit the store probe — otherwise a package
    /// pointed at a local checkpoint would still report its source missing and trigger a
    /// pointless download.
    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fftformer-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let cfg = FFTformerConfiguration(weightsURL: tmp)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty,
                      "explicit weightsURL should satisfy the source")

        // And a path that does NOT exist must fall back to the normal probe.
        let absent = FFTformerConfiguration(
            weightsURL: tmp.appendingPathExtension("nope"))
        XCTAssertEqual(absent.missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN — cancellation

    func testCANGatePreCancelledRun() async {
        // C13: construction is cheap and touches no weights, so this is offline-safe. The entry
        // checkpoint fires before `notLoaded` validation or any image decode.
        let package = FFTformerRestorePackage(configuration: FFTformerConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        // peakActivationBytes (8.0 GB) ≥ 2 GB ⇒ long-run implied; the sub-second exemption is
        // not available to this package.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: FFTformerRestorePackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: FFTformerRestorePackage.manifest,
            posture: .cadence([
                // The real loop: once per tile, before that tile's forward. RunProgress is
                // reported on the same unit, so the cadence is observable rather than asserted.
                .init(phase: .postprocess, unit: .chunk, reportsRunProgress: true),
                // Pre-forward seam: after decode/NHWC conversion, before the tile loop starts.
                .init(phase: .encode, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Manifest sanity

    func testManifestSurfacesAndLicense() {
        let m = FFTformerRestorePackage.manifest

        // Second package on an EXISTING capability — no new capability was introduced.
        XCTAssertEqual(m.capabilities, [.imageRestore])
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .imageRestore)
        XCTAssertEqual(m.surfaces[0].name, "fftformer-deblur")

        // C7 + C8: both layers permissive and declared.
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
    }

    func testFootprintIsSplitAndPlausible() {
        let m = FFTformerRestorePackage.manifest
        let fp = try? XCTUnwrap(m.requirements.footprints.first { $0.quant == .fp32 })
        guard let fp else { return XCTFail("no fp32 footprint declared") }

        // Weights are 16,560,474 params @ fp32 = 66.2 MB; the resident floor must cover them
        // without swallowing the activation (the flat-footprint anti-pattern).
        XCTAssertGreaterThan(fp.residentBytes, 66_000_000)
        XCTAssertLessThan(fp.residentBytes, 500_000_000)

        // The transient is measured at 1080p on the tiled path and must be declared separately.
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
    }

    /// The quant a configuration reports must be the quant a footprint is declared for, or the
    /// governor charges a footprint that does not exist.
    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(FFTformerRestorePackage.manifest.requirements.footprints.map(\.quant))
        for variant in FFTformerVariant.allCases {
            XCTAssertTrue(declared.contains(FFTformerConfiguration(variant: variant).quant),
                          "\(variant) reports a quant with no declared footprint")
        }
    }
}
