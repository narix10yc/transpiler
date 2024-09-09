#include "quench/Polynomial.h"
#include "utils/iocolor.h"

using namespace quench::cas;

int main() {

    auto var0 = std::make_shared<VariableNode>("x[0]")->toPolynomial();
    auto var1 = std::make_shared<VariableNode>("y[1]")->toPolynomial();

    auto const0 = std::make_shared<ConstantNode>(1.2)->toPolynomial();
    auto const1 = std::make_shared<ConstantNode>(1.9)->toPolynomial();

    auto poly1 = var0 + const0 + const1;
    (poly1 * poly1).print(std::cerr) << "\n";

    return 0;
}