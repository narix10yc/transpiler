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


static CircuitGraph getCircuitH1(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("h");
    for (unsigned r = 0; r < repeat; r++)
        for (unsigned q = 0; q < nqubits; q++)
            graph.addGate(mat, {q});
    
    return graph;
} 

static CircuitGraph getCircuitU1(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned r = 0; r < repeat; r++)
        for (unsigned q = 0; q < nqubits; q++)
            graph.addGate(mat, {q});
    
    return graph;
} 

static CircuitGraph getCircuitH2(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("h");
    for (unsigned r = 0; r < repeat; r++) {
        for (unsigned q = 0; q < nqubits; q++) {
            QuantumGate gate(mat, {q});
            gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
            graph.addGate(gate);
        }
    }
    
    return graph;
} 

static CircuitGraph getCircuitU2(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned r = 0; r < repeat; r++) {
        for (unsigned q = 0; q < nqubits; q++) {
            QuantumGate gate(mat, {q});
            gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
            graph.addGate(gate);
        }
    }
    return graph;
} 

static CircuitGraph getCircuitH3(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("h");
    for (unsigned r = 0; r < repeat; r++) {
        for (unsigned q = 0; q < nqubits; q++) {
            QuantumGate gate(mat, {q});
            gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
            gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
            graph.addGate(gate);
        }
    }
    
    return graph;
} 

static CircuitGraph getCircuitU3(int nqubits, int repeat) {
    CircuitGraph graph;
    graph.updateFusionConfig({
            .maxNQubits = 1,
            .maxOpCount = 1,
            .zeroSkippingThreshold = 1e-8
    });

    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned r = 0; r < repeat; r++) {
        for (unsigned q = 0; q < nqubits; q++) {
            QuantumGate gate(mat, {q});
            gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
            gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
            graph.addGate(gate);
        }
    }
    return graph;
} 


int main(int argc, char** argv) {
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::init(""));

    cl::opt<unsigned>
    NQubits("N", cl::desc("number of qubits"), cl::Prefix, cl::Required);

    cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));

    cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));

    cl::opt<bool>
    LoadMatrixInEntry("load-matrix-in-entry", cl::desc(""), cl::init(true));

    cl::opt<bool>
    LoadVectorMatrix("load-vector-matrix", cl::desc(""), cl::init(true));

    cl::opt<unsigned>
    SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));
 
    cl::opt<bool>
    MultiThreaded("multi-thread", cl::desc("enable multi-threading"), cl::init(true));

    cl::opt<std::string>
    WhichGate("gate", cl::desc("which gate"));

    cl::ParseCommandLineOptions(argc, argv);

    CircuitGraph graph;

    if (WhichGate == "h1")
        graph = getCircuitH1(NQubits, 1);
    else if (WhichGate == "u1")
        graph = getCircuitU1(NQubits, 1);
    else if (WhichGate == "h2")
        graph = getCircuitH2(NQubits, 1);
    else if (WhichGate == "u2")
        graph = getCircuitU2(NQubits, 1);
    else if (WhichGate == "h3")
        graph = getCircuitH3(NQubits, 1);
    else if (WhichGate == "u3")
        graph = getCircuitU3(NQubits, 1);
    else {
        std::cerr << RED_FG << "Error: " << RESET
                  << "Unknown gate '" << WhichGate << "'\n";
        return 1;
    }

    CodeGeneratorCPU codeGenerator(outputFilename);
    codeGenerator.config.simd_s = SimdS;
    codeGenerator.config.loadMatrixInEntry = LoadMatrixInEntry;
    codeGenerator.config.loadVectorMatrix = LoadVectorMatrix;
    if (UseF32)
        codeGenerator.config.precision = 32;
    codeGenerator.generate(graph);
    
    return 0;
}