#include "saot/Polynomial.h"
#include "utils/iocolor.h"

using namespace saot::polynomial;

int main() {

    CASContext ctx;

    auto num2 = ctx.getNumerics(2.0);
    auto varx = ctx.getVariable("x");

    auto expr = ctx.createSin(ctx.createMul(num2, varx));
    // auto expr = ctx.createMul(num2, varx);

    expr->print(std::cerr) << "\n";

    expr->derivative("x", ctx)->print(std::cerr) << "\n";
    expr->derivative("x", ctx)->simplify(ctx)->print(std::cerr) << "\n";


    return 0;
}