// #include "openqasm/LegacyParser.h"
// #include "saot/CircuitGraph.h"
// #include "saot/cpu.h"
// #include "utils/iocolor.h"

// #include "llvm/Support/CommandLine.h"

// #include <chrono>
// #include <sstream>

// using QuantumGate = saot::QuantumGate;
// using GateMatrix = saot::GateMatrix;
// using FusionConfig = saot::FusionConfig;
// using CircuitGraph = saot::CircuitGraph;
// using CodeGeneratorCPU = saot::cpu::CodeGeneratorCPU;
// using namespace llvm;
// using namespace IOColor;


// static CircuitGraph getCircuitH1(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("h");
//     for (unsigned r = 0; r < repeat; r++)
//         for (unsigned q = 0; q < nqubits; q++)
//             graph.addGate(mat, {q});
    
//     return graph;
// } 

// static CircuitGraph getCircuitU1(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
//     for (unsigned r = 0; r < repeat; r++)
//         for (unsigned q = 0; q < nqubits; q++)
//             graph.addGate(mat, {q});
    
//     return graph;
// } 

// static CircuitGraph getCircuitH2(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("h");
//     for (unsigned r = 0; r < repeat; r++) {
//         for (unsigned q = 0; q < nqubits; q++) {
//             QuantumGate gate(mat, {q});
//             gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
//             graph.addGate(gate);
//         }
//     }
    
//     return graph;
// } 

// static CircuitGraph getCircuitU2(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
//     for (unsigned r = 0; r < repeat; r++) {
//         for (unsigned q = 0; q < nqubits; q++) {
//             QuantumGate gate(mat, {q});
//             gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
//             graph.addGate(gate);
//         }
//     }
//     return graph;
// } 

// static CircuitGraph getCircuitH3(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("h");
//     for (unsigned r = 0; r < repeat; r++) {
//         for (unsigned q = 0; q < nqubits; q++) {
//             QuantumGate gate(mat, {q});
//             gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
//             gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
//             graph.addGate(gate);
//         }
//     }
    
//     return graph;
// } 

// static CircuitGraph getCircuitU3(int nqubits, int repeat) {
//     CircuitGraph graph;
//     graph.updateFusionConfig({
//             .maxNQubits = 1,
//             .maxOpCount = 1,
//             .zeroSkippingThreshold = 1e-8
//     });

//     auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
//     for (unsigned r = 0; r < repeat; r++) {
//         for (unsigned q = 0; q < nqubits; q++) {
//             QuantumGate gate(mat, {q});
//             gate = gate.lmatmul(QuantumGate(mat, {(q+1) % nqubits}));
//             gate = gate.lmatmul(QuantumGate(mat, {(q+2) % nqubits}));
//             graph.addGate(gate);
//         }
//     }
//     return graph;
// } 


// int main(int argc, char** argv) {
//     cl::opt<std::string>
//     outputFilename("o", cl::desc("output file name"), cl::init(""));

//     cl::opt<unsigned>
//     NQubits("N", cl::desc("number of qubits"), cl::Prefix, cl::Required);

//     cl::opt<std::string>
//     Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));

//     cl::opt<bool>
//     UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));

//     cl::opt<bool>
//     LoadMatrixInEntry("load-matrix-in-entry", cl::desc(""), cl::init(true));

//     cl::opt<bool>
//     LoadVectorMatrix("load-vector-matrix", cl::desc(""), cl::init(true));

//     cl::opt<unsigned>
//     SimdS("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));

//     cl::opt<bool>
//     UsePDEP("use-pdep", cl::desc("use pdep"), cl::init(true));

//     cl::opt<bool>
//     EnablePrefetch("enable-prefetch", cl::desc("enable prefetch"), cl::init(false));

//     cl::opt<bool>
//     AltKernel("alt-kernel", cl::desc("generate alt kernel"), cl::init(false));

//     cl::opt<std::string>
//     WhichGate("gate", cl::desc("which gate"));

//     cl::ParseCommandLineOptions(argc, argv);

//     CircuitGraph graph;

//     if (WhichGate == "h1")
//         graph = getCircuitH1(NQubits, 1);
//     else if (WhichGate == "u1")
//         graph = getCircuitU1(NQubits, 1);
//     else if (WhichGate == "h2")
//         graph = getCircuitH2(NQubits, 1);
//     else if (WhichGate == "u2")
//         graph = getCircuitU2(NQubits, 1);
//     else if (WhichGate == "h3")
//         graph = getCircuitH3(NQubits, 1);
//     else if (WhichGate == "u3")
//         graph = getCircuitU3(NQubits, 1);
//     else {
//         std::cerr << RED_FG << "Error: " << RESET
//                   << "Unknown gate '" << WhichGate << "'\n";
//         return 1;
//     }

//     CodeGeneratorCPU codeGenerator(outputFilename);
//     codeGenerator.config.simd_s = SimdS;
//     codeGenerator.config.loadMatrixInEntry = LoadMatrixInEntry;
//     codeGenerator.config.loadVectorMatrix = LoadVectorMatrix;
//     codeGenerator.config.usePDEP = UsePDEP;
//     codeGenerator.config.enablePrefetch = EnablePrefetch;
//     codeGenerator.config.generateAltKernel = AltKernel;
//     if (UseF32)
//         codeGenerator.config.precision = 32;
//     codeGenerator.generate(graph);
    
//     return 0;
// }

int main() { return 0; }