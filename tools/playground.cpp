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
    ConstantNode c1(3.0);
    
    Polynomial p1 = x.toPolynomial();
    p1 += a.toPolynomial();
    p1 += a.toPolynomial();

    return 0;
}