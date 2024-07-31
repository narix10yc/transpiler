#include "openqasm/parser.h"
#include "quench/CircuitGraph.h"
#include "quench/cpu.h"
#include "utils/iocolor.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>

using FusionConfig = quench::circuit_graph::FusionConfig;
using CircuitGraph = quench::circuit_graph::CircuitGraph;
using CodeGeneratorCPU = quench::cpu::CodeGeneratorCPU;
using namespace llvm;
using namespace Color;

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional, cl::Required);
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::init(""));
    cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));
    cl::opt<unsigned>
    Verbose("verbose", cl::desc("verbose level"), cl::init(1));
    cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));
    cl::opt<unsigned>
    SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));
    cl::opt<bool>
    MultiThreaded("multi-thread", cl::desc("enable multi-threading"), cl::init(true));
    cl::opt<bool>
    InstallTimer("timer", cl::desc("install timer"), cl::init(false));

    // Gate Fusion Category
    cl::OptionCategory GateFusionConfigCategory("Gate Fusion Options", "");
    cl::opt<int>
    FusionLevel("fusion", cl::cat(GateFusionConfigCategory),
            cl::desc("fusion level presets 0 (disable), 1 (two-qubit only), 2 (default), and 3 (aggresive)"),
            cl::init(2));
    cl::opt<int>
    MaxNQubits("max-k", cl::cat(GateFusionConfigCategory),
            cl::desc("maximum number of qubits of gates"), cl::init(0));
    cl::opt<int>
    MaxOpCount("max-op", cl::cat(GateFusionConfigCategory),
            cl::desc("maximum operation count"), cl::init(0));
    cl::opt<double>
    ZeroSkipThreshold("zero-thres", cl::cat(GateFusionConfigCategory),
            cl::desc("zero skipping threshold"), cl::init(1e-8));
    cl::opt<bool>
    AllowMultipleTraverse("allow-multi-traverse", cl::cat(GateFusionConfigCategory),
            cl::desc("allow multiple tile traverse in gate fusion"), cl::init(true));
    cl::opt<bool>
    EnableIncreamentScheme("increment-scheme", cl::cat(GateFusionConfigCategory),
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
    UsePDEP("use-pdep", cl::cat(IRGenerationConfigCategory),
            cl::desc("use pdep (parallel bit deposite)"), cl::init(true));
    cl::opt<bool>
    EnablePrefetch("enable-prefetch", cl::cat(IRGenerationConfigCategory),
            cl::desc("enable prefetch (not tested, recommend off)"), cl::init(false));
    cl::opt<bool>
    AltFormat("alt-format", cl::cat(IRGenerationConfigCategory),
            cl::desc("generate alternating format kernels"), cl::init(false));
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
    auto msg_start = [&]() -> std::string {
        std::stringstream ss;
        ss << "-- ("
           << std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count()
           << " ms) ";
        return ss.str();
    };

    if (Verbose > 0) {
        std::cerr << "-- Input file:  " << inputFilename << "\n";
        std::cerr << "-- Output file: " << outputFilename << "\n";
    }

    openqasm::Parser parser(inputFilename, 0);

    tic = clock::now();
    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";
    auto graph = qasmRoot->toCircuitGraph();
    tok = clock::now();
    std::cerr << msg_start() << "Parsed to CircuitGraph\n";
    if (Verbose > 0)
        graph.displayInfo(std::cerr, 2);

    tic = clock::now();
    if (MaxNQubits > 0 || MaxOpCount > 0) {
        if (MaxNQubits == 0 || MaxOpCount == 0) {
            std::cerr << RED_FG << BOLD << "Argument Error: " << RESET
                      << "need to provide both 'max-k' and 'max-op'\n";
            return 1;
        }
        graph.updateFusionConfig({
                .maxNQubits = MaxNQubits,
                .maxOpCount = MaxOpCount,
                .zeroSkippingThreshold = ZeroSkipThreshold,
            });
    }
    else
        graph.updateFusionConfig(FusionConfig::Preset(FusionLevel));
    graph.getFusionConfig().allowMultipleTraverse = AllowMultipleTraverse;
    graph.getFusionConfig().incrementScheme = EnableIncreamentScheme;

    if (Verbose > 0)
        graph.displayFusionConfig(std::cerr);

    graph.greedyGateFusion();

    tok = clock::now();
    std::cerr << msg_start() << "Greedy gate fusion complete\n";

    if (Verbose > 0)
        graph.relabelBlocks();
    if (Verbose > 2)
        graph.print(std::cerr);
    
    graph.displayInfo(std::cerr, Verbose + 1);

    if (outputFilename != "") {
        tic = clock::now();
        CodeGeneratorCPU codeGenerator(outputFilename);
        // codeGenerator.config.verbose = 3;
        codeGenerator.config.simd_s = SimdS;
        codeGenerator.config.installTimer = InstallTimer;
        codeGenerator.config.multiThreaded = MultiThreaded;
        codeGenerator.config.loadMatrixInEntry = LoadMatrixInEntry;
        codeGenerator.config.loadVectorMatrix = LoadVectorMatrix;
        codeGenerator.config.usePDEP = UsePDEP;
        codeGenerator.config.enablePrefetch = EnablePrefetch;
        codeGenerator.config.generateAltKernel = AltFormat;
        codeGenerator.config.forceDenseKernel = ForceDenseKernel;
        codeGenerator.config.dumpIRToMultipleFiles = DumpIRToMultipleFiles;
        codeGenerator.config.precision = (UseF32 || Precision == "f32") ? 32 : 64;
        
        if (Verbose > 0)
            codeGenerator.displayConfig(std::cerr);

        codeGenerator.generate(graph);
        tok = clock::now();
        std::cerr << msg_start() << "Code generation done\n";
    }
    return 0;
}