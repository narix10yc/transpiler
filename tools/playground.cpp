#include "parse/lexer.h"
#include "qch/ast.h"

using namespace parse;
using namespace qch::ast;

int main(int argc, char** argv) {
    // Lexer lexer(argv[0]);

    CASGraphExpr expr;
    auto x = expr.getVariable("x");
    auto y = expr.getVariable("y");

    auto term = expr.getPow(x, 3.0);
    term = expr.getMul(term, 2.0);

    std::cerr << "Expression: ";
    expr.print(std::cerr);

    std::cerr << "\nDerivative: ";

    auto deriv = expr.computeDerivative("x");
    deriv->print(std::cerr);

    std::cerr << "\nCanonicalize: ";

    deriv->canonicalize(expr.getContext())->print(std::cerr);

    std::cerr << "\n";
    return 0;
}