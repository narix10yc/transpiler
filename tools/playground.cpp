#include "openqasm/parser.h"
#include "quench/Polynomial.h"
#include "quench/simulate.h"
#include "quench/QuantumGate.h"

#include "utils/iocolor.h"
#include "utils/statevector.h"

using namespace Color;
using namespace quench::simulate;
using namespace quench::circuit_graph;
using namespace quench::quantum_gate;
using namespace utils::statevector;

int main(int argc, char** argv) {
    assert(argc > 1);

    openqasm::Parser parser(argv[1], 0);
    auto qasmRoot = parser.parse();
    std::cerr << "qasm AST built\n";
    auto graph = qasmRoot->toCircuitGraph();
    std::cerr << "CircuitGraph built\n";

    // StatevectorComp<double> sv1(graph.nqubits);
    // sv1.zeroState();
    // sv1.randomize();
    // auto sv2 = sv1;

    graph.print(std::cerr);
    graph.displayInfo(std::cerr, 2);
    // for (const auto& block : graph.getAllBlocks()) {
    //     auto gate = block->toQuantumGate();
    //     gate.displayInfo(std::cerr);

    //     applyGeneral(sv1.data, gate.matrix, gate.qubits, sv1.nqubits);
    // }
    // sv1.print(std::cerr) << "\n";

    graph.greedyGateFusion(4);
    graph.print(std::cerr);
    graph.displayInfo(std::cerr, 2);
    // for (const auto& block : graph.getAllBlocks()) {
    //     auto gate = block->toQuantumGate();
    //     gate.displayInfo(std::cerr);

    //     applyGeneral(sv2.data, gate.matrix, gate.qubits, sv2.nqubits);
    // }
    // sv2.print(std::cerr) << "\n";

    return 0;
}