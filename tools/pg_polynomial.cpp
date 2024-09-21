// #include "quench/Polynomial.h"
// #include "quench/parser.h"
// #include "quench/QuantumGate.h"
// #include "quench/CircuitGraph.h"

#include "saot/Polynomial.h"

using namespace saot;

// using namespace quench::cas;
// using namespace quench::ast;
// using namespace quench::quantum_gate;
// using namespace quench::circuit_graph;

int main(int argc, char** argv) {
    // Context ctx;
    // auto mat1 = GateMatrix::FromParameters("u1q", {{"%0"}, {"%1"}, {"%2"}}, ctx);
    // auto mat2 = GateMatrix::FromParameters("u1q", {{"%3"}, {"%4"}, {"%5"}}, ctx);

    // mat1.printMatrix(std::cerr) << "\n";

    // QuantumGate gate1(mat1, {1});
    // QuantumGate gate2(mat1, {2});

    // gate1.lmatmul(gate2).gateMatrix.printMatrix(std::cerr);

    // assert(argc > 1);

    // Parser parser(argv[1]);
    // auto* root = parser.parse();
    // std::cerr << "Recovered:\n";

    // std::ofstream file(std::string(argv[1]) + ".rec");
    // root->print(file);

    // auto graph = root->toCircuitGraph();
    // graph.updateFusionConfig(FusionConfig::Default());
    // graph.greedyGateFusion();

    // graph.getAllBlocks()[0]->quantumGate->displayInfo(std::cerr);
    // graph.displayInfo(std::cerr, 3);

    auto n1 = VariableSumNode::Cosine({0, 1, 2}, 1.2);
    Monomial m1;
    m1.insertMulTerm(n1);

    Polynomial p;
    p.insertMonomial(m1);

    (p * p).print(std::cerr) << "\n";



    return 0;
}