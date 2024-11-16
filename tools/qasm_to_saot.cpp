#include "openqasm/parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "saot/cpu.h"
#include "utils/iocolor.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>

using IRGeneratorConfig = simulation::IRGeneratorConfig;
using AmpFormat = IRGeneratorConfig::AmpFormat;
using namespace IOColor;
using namespace llvm;
using namespace saot;

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional, cl::Required);
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::init(""));
    cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));
    cl::opt<int>
    Verbose("verbose", cl::desc("verbose level"), cl::init(1));
    cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));
    cl::opt<int>
    SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));
    cl::opt<bool>
    MultiThreaded("multi-thread", cl::desc("enable multi-threading"), cl::init(true));
    cl::opt<bool>
    InstallTimer("timer", cl::desc("install timer"), cl::init(false));
    cl::opt<int>
    DebugLevel("debug", cl::desc("IR generation debug level"), cl::init(0));

    // Gate Fusion Category
    cl::OptionCategory GateCPUFusionConfigCategory("Gate Fusion Options", "");
    cl::opt<int>
    FusionLevel("fusion", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("fusion level presets 0 (disable), 1 (two-qubit only), 2 (default), and 3 (aggresive)"),
            cl::init(2));
    cl::opt<int>
    MaxNQubits("max-k", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("maximum number of qubits of gates"), cl::init(0));
    cl::opt<int>
    MaxOpCount("max-op", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("maximum operation count"), cl::init(0));
    cl::opt<double>
    ZeroSkipThreshold("zero-thres", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("zero skipping threshold"), cl::init(1e-8));
    cl::opt<bool>
    AllowMultipleTraverse("allow-multi-traverse", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("allow multiple tile traverse in gate fusion"), cl::init(true));
    cl::opt<bool>
    EnableIncreamentScheme("increment-scheme", cl::cat(GateCPUFusionConfigCategory),
            cl::desc("enable increment fusion scheme"), cl::init(true));

    // IR Generation Category
    cl::OptionCategory IRGenerationConfigCategory("IR Generation Options", "");
    cl::opt<bool>
    LoadMatrixInEntry("load-matrix-in-entry", cl::cat(IRGenerationConfigCategory),
            cl::desc("load matrix in entry"), cl::init(true));
    cl::opt<bool>
    LoadVectorMatrix("load-vector-matrix", cl::cat(IRGenerationConfigCategory),
            cl::desc("load vector matrix"), cl::init(false));
    cl::opt<bool>
    UseFMA("use-fma", cl::cat(IRGenerationConfigCategory),
            cl::desc("use fma (fused multiplication addition)"), cl::init(true));
    cl::opt<bool>
    UseFMS("use-fms", cl::cat(IRGenerationConfigCategory),
            cl::desc("use fms (fused multiplication subtraction)"), cl::init(true));
    cl::opt<bool>
    UsePDEP("use-pdep", cl::cat(IRGenerationConfigCategory),
            cl::desc("use pdep (parallel bit deposite)"), cl::init(true));
    cl::opt<bool>
    EnablePrefetch("enable-prefetch", cl::cat(IRGenerationConfigCategory),
            cl::desc("enable prefetch (not tested, recommend off)"), cl::init(false));
    cl::opt<std::string>
    AmpFormat("amp-format", cl::cat(IRGenerationConfigCategory),
            cl::desc("amplitude format (recommand 'alt')"), cl::init("alt"));
    cl::opt<double>
    ShareMatrixElemThres("share-matrix-elem-thres", cl::cat(IRGenerationConfigCategory),
            cl::desc("share matrix element threshold (set to 0.0 to turn off)"), cl::init(0.0));
    cl::opt<bool>
    ShareMatrixElemUseImmValue("share-matrix-elem-use-imm", cl::cat(IRGenerationConfigCategory),
            cl::desc("use immediate value for shared matrix elements"), cl::init(false));
    cl::opt<bool>
    ForceDenseKernel("force-dense-kernel", cl::cat(IRGenerationConfigCategory),
            cl::desc("force all kernels to be dense"), cl::init(false));
    cl::opt<bool>
    DumpIRToMultipleFiles("dump-ir-to-multiple-files", cl::cat(IRGenerationConfigCategory),
            cl::desc("dump ir to multiple files"), cl::init(false));

    cl::ParseCommandLineOptions(argc, argv);

    using clock = std::chrono::high_resolution_clock;
    auto tic = clock::now();
    auto tok = clock::now();
    auto log = [&]() -> std::ostream& {
        const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count();
        return std::cerr << "-- (" << t_ms << " ms) ";
    };

    if (Verbose > 0) {
        std::cerr << "-- Input file:  " << inputFilename << "\n";
        std::cerr << "-- Output file: " << outputFilename << "\n";
    }

    openqasm::Parser parser(inputFilename, 0);

    // parse and write ast
    tic = clock::now();
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";
    auto graph = qasmRoot->toCircuitGraph();
    tok = clock::now();
    log() << "Parsed to CircuitGraph\n";
    if (Verbose > 2)
        graph.print(std::cerr << "CircuitGraph Before Fusion:\n");
    if (Verbose > 0)
        graph.displayInfo(std::cerr, 2);

    // gate fusion
    tic = clock::now();

    CPUFusionConfig fusionConfig;
    if (MaxNQubits > 0 || MaxOpCount > 0) {
        if (MaxNQubits == 0 || MaxOpCount == 0) {
            std::cerr << RED_FG << BOLD << "Argument Error: " << RESET
                      << "need to provide both 'max-k' and 'max-op'\n";
            return 1;
        }
        fusionConfig.maxNQubits = MaxNQubits;
        fusionConfig.maxOpCount = MaxOpCount;
        fusionConfig.zeroSkippingThreshold = ZeroSkipThreshold;
    }
    else
        fusionConfig = CPUFusionConfig::Preset(FusionLevel);
    fusionConfig.allowMultipleTraverse = AllowMultipleTraverse;
    fusionConfig.incrementScheme = EnableIncreamentScheme;

    if (Verbose > 0)
        fusionConfig.display(std::cerr);

    saot::applyCPUGateFusion(fusionConfig, graph);

    tok = clock::now();
    log() << "Gate fusion complete\n";


    if (Verbose > 2) {
        graph.relabelBlocks();
        graph.print(std::cerr << "CircuitGraph After Fusion:\n");
    }
    if (Verbose > 0)
        graph.displayInfo(std::cerr, Verbose + 1);
    // write to file if provided
    if (outputFilename != "") {
        tic = clock::now();
        const auto config = CodeGeneratorCPUConfig {
            .multiThreaded = MultiThreaded,
            .installTimer = InstallTimer,
            .irConfig = IRGeneratorConfig {
                .simd_s = SimdS,
                .precision = (UseF32 || Precision == "f32") ? 32 : 64,
                .ampFormat = (AmpFormat == "sep") ? IRGeneratorConfig::SepFormat : IRGeneratorConfig::AltFormat,
                .useFMA = UseFMA,
                .useFMS = UseFMS,
                .usePDEP = UsePDEP,
                .loadMatrixInEntry = LoadMatrixInEntry,
                .loadVectorMatrix = LoadVectorMatrix,
                .forceDenseKernel = ForceDenseKernel,
                .zeroSkipThres = ZeroSkipThreshold,
                .shareMatrixElemThres = ShareMatrixElemThres,
                .shareMatrixElemUseImmValue = ShareMatrixElemUseImmValue
            }
        };

        CodeGeneratorCPU codeGenerator(config, outputFilename);
        if (Verbose > 0) {
            codeGenerator.displayConfig(Verbose, std::cerr);
            config.irConfig.checkConfliction(std::cerr);
        }

        codeGenerator.generate(graph, DebugLevel);

        tok = clock::now();
        log() << "Code generation done\n";
    }
    return 0;
}