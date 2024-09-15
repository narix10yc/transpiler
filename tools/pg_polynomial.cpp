#include "quench/Polynomial.h"
#include "quench/parser.h"
#include "quench/QuantumGate.h"

using namespace quench::cas;
using namespace quench::ast;
using namespace quench::quantum_gate;

int main(int argc, char** argv) {

    // auto var0 = (new VariableNode("x[0]"))->toPolynomial();
    // auto var1 = (new VariableNode("y[1]"))->toPolynomial();

    // auto const0 = (new ConstantNode(1.2))->toPolynomial();
    // auto const1 = (new ConstantNode(1.9))->toPolynomial();

    // auto poly1 = var0 + const0 + const1;
    // (poly1 * poly1).print(std::cerr) << "\n";

    // assert(argc > 1);

    // Parser parser(argv[1]);
    // auto* root = parser.parse();
    // // std::cerr << "Recovered:\n";

    // std::ofstream file(std::string(argv[1]) + ".rec");
    // root->print(file);

    Context ctx;
    auto mat1 = GateMatrix::FromParameters("u1q", {{"%0"}, {"%1"}, {"%2"}}, ctx);
    auto mat2 = GateMatrix::FromParameters("u1q", {{"%3"}, {"%4"}, {"%5"}}, ctx);

    mat1.printMatrix(std::cerr) << "\n";

    QuantumGate gate1(mat1, {1});
    QuantumGate gate2(mat1, {2});

    gate1.lmatmul(gate2).gateMatrix.printMatrix(std::cerr);


    

    return 0;
}