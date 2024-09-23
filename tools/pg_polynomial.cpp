#include "quench/parser.h"
#include "quench/QuantumGate.h"
#include "quench/CircuitGraph.h"

#include "saot/Polynomial.h"

using namespace saot;

using namespace quench::ast;
using namespace quench::quantum_gate;
using namespace quench::circuit_graph;

int main(int argc, char** argv) {
    std::vector<std::pair<int, double>> varValues {
        {0, 1.1}, {1, 0.4}, {2, 0.1}, {3, -0.3}, {4, -0.9}, {5, 1.9}};

    // auto mat1 = GateMatrix::FromParameters("u1q", std::vector<GateParameter>{GateParameter(0), GateParameter(1), GateParameter(2)});
    // auto mat2 = GateMatrix::FromParameters("u1q", std::vector<GateParameter>{GateParameter(3), GateParameter(4), GateParameter(5)});

    // mat1.printMatrix(std::cerr) << "\n";

    // QuantumGate gate1(mat1, {1});
    // QuantumGate gate2(mat2, {1});

    // auto gate = gate1.lmatmul(gate2);
    // gate.gateMatrix.printMatrix(std::cerr) << "\n";



    // for (auto& P : gate.gateMatrix.pData())
    //     P.simplify(varValues);
    

    // gate.gateMatrix.printMatrix(std::cerr);

    assert(argc > 1);

    Parser parser(argv[1]);
    auto* root = parser.parse();
    std::cerr << "Recovered:\n";

    std::ofstream file(std::string(argv[1]) + ".rec");
    root->print(file);

    auto graph = root->toCircuitGraph();
    graph.updateFusionConfig(FusionConfig::Default());
    graph.greedyGateFusion();

    auto* fusedGate = graph.getAllBlocks()[0]->quantumGate.get();

    fusedGate->gateMatrix.printMatrix(std::cerr) << "\n";

    for (auto& P : fusedGate->gateMatrix.pData()) {
        P.removeSmallMonomials();
    }
    fusedGate->gateMatrix.printMatrix(std::cerr) << "\n";

    for (auto& P : fusedGate->gateMatrix.pData())
        P.simplify(varValues);
    fusedGate->gateMatrix.printMatrix(std::cerr);

    // Monomial m1;
    // m1.insertMulTerm(VariableSumNode::Cosine({0, 2}, 1.2));
    // m1.insertExpiVar(3, false);
    // m1.print(std::cerr) << "\n";

    // m1.insertExpiVar(3, true);
    // m1.print(std::cerr) << "\n";
    // Polynomial p;
    // p.insertMonomial(m1);

    // p.print(std::cerr) << "\n";

    // p.simplify({{0, 0.77}, {2, 1.14}, {3, 4.0}});
    // p.print(std::cerr) << "\n";

    return 0;
}