#include "openqasm/parser.h"
// #include "quench/parser.h"
#include "quench/GateMatrix.h"

// using namespace quench::ast;
// using namespace quench::cas;

int main(int argc, char** argv) {
    openqasm::Parser parser(argv[1], 0);
    auto qasmRoot = parser.parse();
    std::cerr << "qasm AST built\n";

    auto graph = qasmRoot->toCircuitGraph();

    std::cerr << "CircuitGraph built\n";

    graph.print(std::cerr);

    // Parser parser("../examples/simple.qch");
    // auto root = parser.parse();

    return 0;
}