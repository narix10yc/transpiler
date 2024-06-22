#include "openqasm/parser.h"
// #include "quench/parser.h"
#include "quench/GateMatrix.h"

// using namespace quench::ast;
// using namespace quench::cas;

using namespace quench::circuit_graph;

int main(int argc, char** argv) {
    openqasm::Parser parser(argv[1], 0);
    auto qasmRoot = parser.parse();
    std::cerr << "qasm AST built\n";

    auto graph = qasmRoot->toCircuitGraph();

    std::cerr << "CircuitGraph built\n";

    std::cerr << "Before Fusion: " << graph.countBlocks() << " blocks\n";
    graph.print(std::cerr, 2) << "\n";
    graph.displayInfo(std::cerr, 2) << "\n";

    // graph.updateTileUpward();
    // std::cerr << "upward\n";
    // graph.print(std::cerr, 2) << "\n";

    // graph.updateTileDownward();
    // std::cerr << "downward\n";
    // graph.print(std::cerr, 2);

    // graph.fuseToTwoQubitGates();

    for (unsigned maxNqubits = 2; maxNqubits < 5; maxNqubits++) {
        graph.greedyGateFusion(maxNqubits);
        std::cerr << "After Greedy Fusion " << maxNqubits << ":\n";
        graph.print(std::cerr, 2);
        graph.displayInfo(std::cerr, 2) << "\n";
    }



    // graph.greedyGateFusion(4);
    // std::cerr << "After Fusion 2: " << graph.countBlocks() << " blocks\n";
    // graph.print(std::cerr, 2);

    // graph.applyInOrder([](GateBlock* block) {
        // std::cerr << "calling block " << block->id << "\n";
    // });

    // Parser parser("../examples/simple.qch");
    // auto root = parser.parse();

    return 0;
}