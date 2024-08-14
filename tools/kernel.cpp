#include "openqasm/parser.h"
#include "quench/CircuitGraph.h"
#include "quench/cpu.h"
#include "utils/iocolor.h"
#include <functional>

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>

using QuantumGate = quench::quantum_gate::QuantumGate;
using GateMatrix = quench::quantum_gate::GateMatrix;
using FusionConfig = quench::circuit_graph::FusionConfig;
using CircuitGraph = quench::circuit_graph::CircuitGraph;
using IRGeneratorConfig = simulation::IRGeneratorConfig;
using AmpFormat = IRGeneratorConfig::AmpFormat;

using namespace quench::cpu;
using namespace llvm;
using namespace Color;


static CircuitGraph& getCircuitH1(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("h");
    for (unsigned q = 0; q < nqubits; q++)
        graph.addGate(mat, {q});
    
    return graph;
} 

static CircuitGraph& getCircuitU1(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned q = 0; q < nqubits; q++)
        graph.addGate(mat, {q});
    
    return graph;
} 

static CircuitGraph& getCircuitH2(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("h");
    for (unsigned q = 0; q < nqubits; q++) {
        QuantumGate gate(mat, {q});
        gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
        graph.addGate(gate);
    }
    
    return graph;
} 

static CircuitGraph& getCircuitU2(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned q = 0; q < nqubits; q++) {
        QuantumGate gate(mat, {q});
        gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
        graph.addGate(gate);
    }
    
    return graph;
} 

static CircuitGraph& getCircuitH3(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("h");
    for (unsigned q = 0; q < nqubits; q++) {
        QuantumGate gate(mat, {q});
        gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
        gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
        graph.addGate(gate);
    }
    
    return graph;
} 

static CircuitGraph& getCircuitU3(CircuitGraph& graph, int nqubits) {
    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    for (unsigned q = 0; q < nqubits; q++) {
        QuantumGate gate(mat, {q});
        gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
        gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
        graph.addGate(gate);
    }

    return graph;
} 


int main(int argc, char** argv) {
    cl::opt<unsigned>
    NQubits("N", cl::desc("number of qubits"), cl::Prefix, cl::Required);
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
    UseFMA("use-fma", cl::cat(IRGenerationConfigCategory),
            cl::desc("use fma (fused multiplicatio addition)"), cl::init(true));
    cl::opt<bool>
    UseFMS("use-fms", cl::cat(IRGenerationConfigCategory),
            cl::desc("use fms (fused multiplicatio subtraction)"), cl::init(true));
    cl::opt<bool>
    UsePDEP("use-pdep", cl::cat(IRGenerationConfigCategory),
            cl::desc("use pdep (parallel bit deposite)"), cl::init(true));
    cl::opt<bool>
    EnablePrefetch("enable-prefetch", cl::cat(IRGenerationConfigCategory),
            cl::desc("enable prefetch (not tested, recommend off)"), cl::init(false));
    cl::opt<std::string>
    AmpFormat("amp-format", cl::cat(IRGenerationConfigCategory),
            cl::desc("amplitude format"), cl::init("alt"));
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

    cl::opt<bool>
    FullKernel("full", cl::desc("generate alt kernel"), cl::init(false));

    cl::opt<std::string>
    WhichGate("gate", cl::desc("which gate"));

    cl::ParseCommandLineOptions(argc, argv);

    CircuitGraph graph;
    graph.updateFusionConfig(FusionConfig::Disable());

    const auto wrapper = [&](CircuitGraph& (*f)(CircuitGraph&, int)) {
        if (FullKernel) {
            for (int nqubits = 8; nqubits <= NQubits; nqubits += 2)
                f(graph, nqubits);
        } else {
            f(graph, NQubits);
        }
    };

    if (WhichGate == "h1")
        wrapper(getCircuitH1);
    else if (WhichGate == "u1")
        wrapper(getCircuitU1);
    else if (WhichGate == "h2")
        wrapper(getCircuitH2);
    else if (WhichGate == "u2")
        wrapper(getCircuitU2);
    else if (WhichGate == "h3")
        wrapper(getCircuitH3);
    else if (WhichGate == "u3")
        wrapper(getCircuitU3);
    else {
        std::cerr << RED_FG << "Error: " << RESET
                  << "Unknown gate '" << WhichGate << "'\n";
        return 1;
    }

    const auto config = CodeGeneratorCPUConfig {
        .multiThreaded = MultiThreaded,
        .installTimer = InstallTimer,
        .irConfig = IRGeneratorConfig {
            .simd_s = SimdS,
            .precision = (UseF32 || Precision == "f32") ? 32 : 64,
            .ampFormat = (AmpFormat == "sep") ? AmpFormat::Sep : AmpFormat::Alt,
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

    codeGenerator.generate(graph, DebugLevel, true); // force in order
    
    return 0;
}
