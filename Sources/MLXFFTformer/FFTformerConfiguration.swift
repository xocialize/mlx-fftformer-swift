import Foundation
import MLXToolKit

/// An FFTformer checkpoint this package can load.
///
/// Only the GoPro checkpoint exists upstream today — `kkkls/FFTformer` commits exactly one file,
/// `pretrain_model/fftformer_GoPro.pth`. The enum exists so a RealBlur-trained sibling (upstream
/// ships `options/train/Realblur.yml` but no matching weights) can be added later without changing
/// the package shape.
public enum FFTformerVariant: String, Codable, Sendable, CaseIterable {
    /// GoPro-trained motion deblur — the released checkpoint. GoPro 34.21 dB.
    case gopro

    public var repo: String {
        switch self {
        case .gopro: return "mlx-community/FFTformer-GoPro-fp32"
        }
    }

    /// fp32, deliberately — measured, not inherited from the "conv models ship fp16" rule.
    ///
    /// bf16 is disqualified (cosine 0.9879 / 30.09 dB vs the fp32 reference: a *dtype* error the
    /// same order as the model's own 34.21 dB task signal). fp16 is viable at 50.13 dB but was not
    /// chosen — the model is only 66 MB, so halving it buys little while leaving the dtype error
    /// just ~16 dB below the signal. See PORT-STATUS.md "Publish dtype".
    public var quant: Quant {
        switch self {
        case .gopro: return .fp32
        }
    }
}

/// Init-time configuration for `FFTformerRestorePackage` (C9).
public struct FFTformerConfiguration: PackageConfiguration, ModelStorable {
    public var variant: FFTformerVariant

    /// Where downloadable weights are materialized. Set by the engine from its `ModelStore`;
    /// `nil` → the default hub cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    /// Explicit local weights file, bypassing the store entirely. Escape hatch for parity work and
    /// for running against a locally converted checkpoint before it is published.
    public var weightsURL: URL?

    public init(variant: FFTformerVariant = .gopro,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}

/// `QuantConfigured` (engine 1.14): charge the declared `QuantFootprint` for the selected variant
/// rather than the largest-that-fits heuristic.
extension FFTformerConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

/// `WeightSourcing` (engine 0.19.0 / contract 1.24): declare what a fresh machine would fetch, so
/// the ENGINE materializes it before `load()` and `load()` only loads.
extension FFTformerConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights",
                      repo: variant.repo,
                      revision: nil,
                      matching: ["model.safetensors"])]
    }

    /// Overridden per MS-2 guidance: honor the explicit `weightsURL` escape hatch FIRST, then fall
    /// back to the default store probe. Without this, a configuration pointed at a local file would
    /// still report its source as missing and trigger a pointless download.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
