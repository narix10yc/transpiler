// #include "quench/parser.h"
#include "qch/GateMatrix.h"

// using namespace quench::ast;

using namespace qch::ir;

int main() {
    // Parser parser("../examples/simple.qch");
    // auto root = parser.parse();

    VariableNode x("x");
    VariableNode y("y");
    VariableNode a("a");
    CosineNode cos1(std::make_shared<VariableNode>(y));
    ConstantNode c1(3.0);
    
    Polynomial p1 = x.toPolynomial();
    p1 += a.toPolynomial();
    p1 *= a.toPolynomial();
    p1 *= p1;
    p1 += cos1.toPolynomial();
    p1 *= cos1.toPolynomial();
    std::cerr << "LaTeX: ";
    p1.printLaTeX(std::cerr) << "\n";
    return 0;
}