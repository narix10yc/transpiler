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
    outputFilename("o", cl::desc("output file name"), cl::init(""));

    cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));

    cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));

    cl::opt<unsigned>
    SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));
 
    cl::opt<bool>
    MultiThreaded("multi-thread", cl::desc("enable multi-threading"), cl::init(true));

    cl::ParseCommandLineOptions(argc, argv);

    // parse and write ast
    std::cerr << "-- qasm AST built\n";
    auto graph = CircuitGraph();

    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });
    
    CodeGeneratorCPU codeGenerator();



    codeGenerator.generate(graph);
    
    return 0;
}