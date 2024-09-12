#include "quench/Polynomial.h"
#include "quench/parser.h"

using namespace quench::cas;
using namespace quench::ast;

int main(int argc, char** argv) {

    auto var0 = (new VariableNode("x[0]"))->toPolynomial();
    auto var1 = (new VariableNode("y[1]"))->toPolynomial();

    auto const0 = (new ConstantNode(1.2))->toPolynomial();
    auto const1 = (new ConstantNode(1.9))->toPolynomial();

    auto poly1 = var0 + const0 + const1;
    (poly1 * poly1).print(std::cerr) << "\n";

    assert(argc > 1);

    Parser parser(argv[1]);
    auto* root = parser.parse();
    std::cerr << "Recovered:\n";
    root->print(std::cerr);

    return 0;
}