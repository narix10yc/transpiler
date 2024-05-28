#include "qch/cas.h"

using namespace qch::cas;

int main(int argc, char** argv) {
    auto x = std::make_shared<Variable>("x");
    auto y = std::make_shared<Variable>("y");
    auto cos1 = std::make_shared<Cosine>(x);

    Power p1(cos1, 1);
    Power p2(y, 3);
    Power p3(x, 4);


    std::cerr << "== Test Monomial * Power: == ";

    Monomial m1({p1, p2}, 2);
    std::cerr << "\nm1: "; m1.print(std::cerr);

    Power p4(cos1, 2);
    std::cerr << "\np4: "; p4.print(std::cerr);
    std::cerr << "\nm1 * p4: "; (m1 * p4).print(std::cerr);
    std::cerr << "\n\n";


    std::cerr << "== Test Monomial * Monomial: == ";

    std::cerr << "\nm1: "; m1.print(std::cerr);

    Monomial m2({p2, p3}, -1.0);
    std::cerr << "\nm2: "; m2.print(std::cerr);
    std::cerr << "\nm1 * m2: "; (m1 * m2).print(std::cerr);
    std::cerr << "\n\n";


    std::cerr << "== Test Polynomial * Monomial: == ";

    Polynomial poly1({m1, m2});
    std::cerr << "\npoly1: "; poly1.print(std::cerr);

    std::cerr << "\nm2: "; m2.print(std::cerr);
    auto poly2 = poly1 * m2;
    poly2 = poly2.sortAndSimplify();
    std::cerr << "\npoly1 * m2: "; poly2.print(std::cerr);
    std::cerr << "\n\n";


    std::cerr << "== Test Polynomial * Polynomial: == ";

    std::cerr << "\npoly1: "; poly1.print(std::cerr);

    std::cerr << "\npoly1 * poly1: "; (poly1 * poly1).sortAndSimplify().print(std::cerr);
    std::cerr << "\n\n";


    std::cerr << "\n(poly2 * poly2): "; (poly2 * poly2).print(std::cerr);

    return 0;
}