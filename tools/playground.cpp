#include "openqasm/parser.h"
#include "quench/CircuitGraph.h"
#include "quench/cpu.h"
#include "quench/simulate.h"
#include "utils/iocolor.h"
#include "utils/statevector.h"
// #include "utils/half.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>
#include <cmath>

using namespace utils::statevector;
using namespace quench::simulate;
using QuantumGate = quench::quantum_gate::QuantumGate;
using GateMatrix = quench::quantum_gate::GateMatrix;
using namespace llvm;
using namespace Color;

// #include <immintrin.h>

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional);

    cl::opt<unsigned>
    NQubits("N", cl::desc("number of qubits"), cl::Prefix, cl::Required);

    cl::ParseCommandLineOptions(argc, argv);

    StatevectorComp<double> sv64(NQubits);
    StatevectorComp<float> sv32(NQubits);

    sv64.randomize();
    for (unsigned i = 0; i < sv64.N; i++)
        sv32.data[i] = sv64.data[i];

    std::cerr << "Norm of f64 before: " << sv64.norm() << "\n";
    std::cerr << "Norm of f32 before: " << sv32.norm() << "\n\n";

    openqasm::Parser parser(inputFilename, 0);

    // parse and write ast
    // auto qasmRoot = parser.parse();
    // std::cerr << "-- qasm AST built\n";
    // auto graph = qasmRoot->toCircuitGraph();

    // graph.greedyGateFusion();

    // const auto allBlocks = graph.getAllBlocks();
    // for (const auto* block : allBlocks) {
    //     const auto& gate = block->quantumGate;
    //     applyGeneral<double>(sv64.data, gate->gateMatrix, gate->qubits, NQubits);
    //     applyGeneral<float>(sv32.data, gate->gateMatrix, gate->qubits, NQubits);
    // }


    const auto matH = GateMatrix::FromName("h"); 

    for (unsigned q = 0; q < NQubits; q++) {
        applyGeneral<double>(sv64.data, matH, { q }, NQubits);
        applyGeneral<float>(sv32.data, matH, { q }, NQubits);
        for (unsigned qq = q+1; qq < NQubits; qq++) {
            double lambd = M_PI / static_cast<double>(1 << (qq - q));
            const auto matCP = GateMatrix::FromName("cp", {lambd});
            applyGeneral<double>(sv64.data, matCP, {q, qq}, NQubits);
            applyGeneral<float>(sv32.data, matCP, {q, qq}, NQubits);
        }
    }

    std::cerr << "Norm of f64 after: " << sv64.norm() << "\n";
    std::cerr << "Norm of f32 after: " << sv32.norm() << "\n";

    double infidality = 0.0;
    for (uint64_t i = 0; i < sv64.N; i++) {
        double dif_re = sv64.data[i].real() - sv32.data[i].real();
        double dif_im = sv64.data[i].imag() - sv32.data[i].imag();

        infidality += dif_re * dif_re + dif_im * dif_im;
    }
    std::cerr << "infidality = " << infidality << "\n";

    return 0;
}