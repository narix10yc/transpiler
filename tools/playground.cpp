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
    // assert(argc > 1);

    // openqasm::Parser parser(argv[1], 0);
    // auto qasmRoot = parser.parse();
    // std::cerr << "qasm AST built\n";
    // auto graph = qasmRoot->toCircuitGraph();
    // std::cerr << "CircuitGraph built\n";

    // StatevectorComp<double> sv1(graph.nqubits);
    // sv1.zeroState();
    // sv1.randomize();
    // auto sv2 = sv1;
    // auto allBlocks = graph.getAllBlocks();
    // std::vector<GateNode*> gates;
    // std::vector<unsigned> qubits;

    // std::cerr << "Before Fusion: " << graph.countBlocks() << " blocks\n";
    // // graph.print(std::cerr, 2) << "\n";
    // graph.displayInfo(std::cerr, 2) << "\n";
    // for (const auto& block : allBlocks) {
    //     gates.clear();
    //     block->applyInOrder([&gates](GateNode* g) { gates.push_back(g); });

    //     std::cerr << CYAN_FG << BOLD << "gates in block " << block->id << ": ";
    //     for (const auto* gate : gates)
    //         std::cerr << gate->id << ",";
    //     std::cerr << RESET << "\n";

    //     for (const auto& gate : gates) {
    //         qubits.clear();
    //         for (const auto& data : gate->dataVector)
    //             qubits.push_back(data.qubit);

    //         applyGeneral<double>(sv1.data, gate->gateMatrix, qubits, sv1.nqubits);
    //     }
    // }

    // for (int m = 2; m < 6; m++) {
    //     graph.greedyGateFusion(m);
    //     std::cerr << "After Greedy Fusion " << m << ":\n";
    //     // graph.print(std::cerr, 2);
    //     graph.displayInfo(std::cerr, 2) << "\n";
    // }
    // allBlocks = graph.getAllBlocks();

    // sv2.print(std::cerr);
    // for (const auto& block : allBlocks) {
    //     gates.clear();
    //     block->applyInOrder([&gates](GateNode* g) { gates.push_back(g); });
    //     std::cerr << CYAN_FG << BOLD << "gates in block " << block->id << ": ";
    //     for (const auto* gate : gates)
    //         std::cerr << gate->id << ",";
    //     std::cerr << RESET << "\n";

    //     for (const auto& gate : gates) {
    //         qubits.clear();
    //         for (const auto& data : gate->dataVector)
    //             qubits.push_back(data.qubit);

    //         applyGeneral<double>(sv2.data, gate->gateMatrix, qubits, sv2.nqubits);
    //     }
    // }
    // sv1.print(std::cerr) << "\n";
    // sv2.print(std::cerr);

    auto m1 = GateMatrix::FromName("u3", {M_PI * 0.5, 0.0, M_PI});
    m1.printMatrix(std::cerr) << "\n";

    QuantumGate gate1(m1, {2});
    QuantumGate gate2(m1, {3});

    gate2.lmatmul(gate1).lmatmul({m1, 4}).displayInfo(std::cerr);

    return 0;
}