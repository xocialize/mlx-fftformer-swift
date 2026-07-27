// swift-tools-version: 6.2
import PackageDescription

// mlx-fftformer-swift — FFTformer motion deblurring for MLXEngine. ONE repo, TWO products:
//   • FFTformerMLXCore — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXFFTformer    — the MLXEngine `imageRestore` ModelPackage over that core
// Mirrors the mlx-nafnet-swift layout deliberately: FFTformer is a SECOND package on the existing
// `imageRestore` capability (selected by PackageID alongside NAFNet's .goproWidth64), not a new
// capability, so it needs no engine contract change.
//
// Upstream: kkkls/FFTformer (MIT, weights committed in-repo). See PORT-STATUS.md.
//
// NOTE: no MLXFFT dependency. `rfft2`/`irfft2` live in the main MLX module now; the MLXFFT
// re-exports are deprecated ("now available in the main MLX module").
let package = Package(
    name: "mlx-fftformer-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "FFTformerMLXCore", targets: ["FFTformerMLXCore"]),
        .library(name: "MLXFFTformer", targets: ["MLXFFTformer"]),
        .executable(name: "fftformer-gate", targets: ["FFTformerGate"]),
        .executable(name: "fftformer-validate", targets: ["FFTformerValidate"]),
    ],
    dependencies: [
        // 0.38.0 = contract 1.29.0, and the first tag carrying `licenseEnforcement`
        // (contract 1.28.0 / v0.37.0), which the validate target sets to `.blocking` to match
        // how Forge constructs the engine in production.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.38.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "FFTformerMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXFFTformer",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "FFTformerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "FFTformerMLXTests",
            dependencies: [
                "FFTformerMLXCore",
                "MLXFFTformer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            resources: [
                .copy("Resources/goldens"),
            ]
        ),
        // Drives the package through the REAL MLXServeEngine and reports the authoritative split
        // footprint via MLXEngineTestKit — same harness and same phys_footprint metric the archived
        // validation app used. The gate's `--bench` reads MLX-pool memory, which under-reads the
        // admission basis by ~2.7x (the BiRefNet re-baseline).
        .executableTarget(
            name: "FFTformerValidate",
            dependencies: [
                "MLXFFTformer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/Validate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Parity gates that need a real Metal context live HERE, not in the test target — the SPM
        // test product's metallib is unreliable (see mlx-swift-integration/swift-port-parity.md).
        .executableTarget(
            name: "FFTformerGate",
            dependencies: [
                "FFTformerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
