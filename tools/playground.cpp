#include "qch/cas.h"

using namespace qch::cas;

int main(int argc, char** argv) {
    auto x = std::make_shared<Variable>("x");
    auto y = std::make_shared<Variable>("y");
    auto cos1 = std::make_shared<Cosine>(x);

    auto pow1 = std::make_shared<Power>(x, 2);
    auto pow2 = std::make_shared<Power>(x, 4);
    auto pow3 = std::make_shared<Power>(cos1, 2);

    auto m1 = std::make_shared<Monomial>(pow1, 3.0);

    auto poly1 = pow2->toPolynomial();
    auto poly2 = pow1->toPolynomial();
    auto poly3 = pow3->toPolynomial();

    std::cerr << "poly1 = ";
    poly1.print(std::cerr);

    std::cerr << "\n\npoly2 = ";
    poly2.print(std::cerr);

    std::cerr << "\n\npoly3 = ";
    poly3.print(std::cerr);

    auto poly4 = poly2 + poly3;
    auto poly5 = poly3 + poly2;

    poly4.sort();
    poly5.sort();

    std::cerr << "\n\npoly2 + poly3 = ";
    poly4.print(std::cerr);

    std::cerr << "\n\npoly3 + poly2 = ";
    (poly5 + poly3).print(std::cerr);


    std::cerr << "\n";

    return 0;
}