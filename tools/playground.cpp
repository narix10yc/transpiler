#include "openqasm/parser.h"
#include "quench/CircuitGraph.h"
#include "quench/cpu.h"
#include "utils/iocolor.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>

using QuantumGate = quench::quantum_gate::QuantumGate;
using GateMatrix = quench::quantum_gate::GateMatrix;
using FusionConfig = quench::circuit_graph::FusionConfig;
using CircuitGraph = quench::circuit_graph::CircuitGraph;
using CodeGeneratorCPU = quench::cpu::CodeGeneratorCPU;
using namespace llvm;
using namespace Color;


int main(int argc, char** argv) {
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::init(""));

    // cl::opt<unsigned>
    // TargetQubit1("Q", cl::desc("target qubit 1"), cl::Prefix, cl::Required);

    // cl::opt<unsigned>
    // TargetQubit2("R", cl::desc("target qubit 2"), cl::Prefix, cl::Required);

    cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));

    cl::opt<unsigned>
    SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));

    cl::ParseCommandLineOptions(argc, argv);

    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    CodeGeneratorCPU codeGenerator(outputFilename);
    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    auto gate = QuantumGate(mat, { 7 });
    gate = gate.lmatmul({ mat , { 8 }});
    gate = gate.lmatmul({ mat , { 9 }});
    graph.addGate(gate);

    codeGenerator.config.s = SimdS;
    if (UseF32)
        codeGenerator.config.precision = 32;
    codeGenerator.generate(graph, 100);
    
    return 0;
}