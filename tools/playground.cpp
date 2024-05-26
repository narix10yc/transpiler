#include "qch/cas.h"

using namespace qch::cas;

int main(int argc, char** argv) {
    CASContext ctx;
    auto x = ctx.getVariable("x");
    auto cosx = ctx.getCosine(x); 
    auto x2 = ctx.getPower(x, 2);

    auto monomial = ctx.getMonomial(2.0, {ctx.getPower(cosx), x2, ctx.getPower(x)});
    auto p1 = monomial->canonicalize(ctx);
    
    p1->print(std::cerr);
    std::cerr << "\n";
    std::cerr << "There are " << ctx.count() << " nodes in context\n";
    
    Polynomial p2 { *p1 };

    p2.print(std::cerr);
    std::cerr << "\n";
    std::cerr << "There are " << ctx.count() << " nodes in context\n";

    (p2 + (*p1)).print(std::cerr);
    std::cerr << "\n";
    std::cerr << "There are " << ctx.count() << " nodes in context\n";

    p2.print(std::cerr);
    std::cerr << "\n";
    std::cerr << "There are " << ctx.count() << " nodes in context\n";
    return 0;
}