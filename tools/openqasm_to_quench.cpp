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
    InstallTimer("timer", cl::desc("install timer"), cl::init(false));

    cl::opt<unsigned>
    MaxNQubits("max-k", cl::desc("maximum number of qubits of gates"), cl::init(0));

    cl::opt<unsigned>
    MaxOpCount("max-op", cl::desc("maximum operation count"), cl::init(0));

    cl::opt<double>
    ZeroSkipThreshold("zero-thres", cl::desc("zero skipping threshold"), cl::init(1e-8));

    cl::opt<int>
    FusionLevel("fusion", cl::desc("fusion level. Presets are "
            "0 (disable), 1 (two-qubit only), 2 (default), and 3 (aggresive)"),
            cl::init(2));

    cl::opt<bool>
    MultiThreaded("multi-thread", cl::desc("enable multi-threading"), cl::init(true));

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
                .maxNQubits = static_cast<int>(MaxNQubits),
                .maxOpCount = static_cast<int>(MaxOpCount),
                .zeroSkippingThreshold = ZeroSkipThreshold
            });
    }
    else {
        graph.updateFusionConfig(FusionConfig::Preset(FusionLevel));
    }

    if (Verbose > 0)
        graph.displayFusionConfig(std::cerr);

    unsigned maxK0 = graph.getFusionConfig().maxNQubits;
    for (unsigned maxK = 2; maxK <= maxK0; maxK++) {
        graph.getFusionConfig().maxNQubits = maxK;
        graph.greedyGateFusion();
    }
    tok = clock::now();
    std::cerr << msg_start() << "Greedy gate fusion complete\n";

    if (Verbose > 1) {
        graph.relabelBlocks();
        graph.displayInfo(std::cerr, 3);
        graph.print(std::cerr);
    }
    else if (Verbose > 0) {
        graph.displayInfo(std::cerr, 2);
    }

    if (outputFilename != "") {
        tic = clock::now();
        CodeGeneratorCPU codeGenerator(outputFilename);
        codeGenerator.config_s(SimdS);
        codeGenerator.config_timer(InstallTimer);
        codeGenerator.config_multiThreaded(MultiThreaded);
        if (UseF32 || Precision == "f32")
            codeGenerator.config_precision(32);
        else
            codeGenerator.config_precision(64);
        
        if (Verbose > 0)
            codeGenerator.displayConfig(std::cerr);

        codeGenerator.generate(graph);
        tok = clock::now();
        std::cerr << msg_start() << "Code generation done\n";
    }
    return 0;
}