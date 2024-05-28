#include "qch/cas.h"

using namespace qch::cas;

Polynomial Constant::toPolynomial() const {
    return { Monomial(std::make_shared<Constant>(*this)) };
}

Polynomial Variable::toPolynomial() const {
    return { Monomial(std::make_shared<Variable>(*this)) };
}

Polynomial Cosine::toPolynomial() const {
    return { Monomial(std::make_shared<Cosine>(*this)) };
}

Polynomial Sine::toPolynomial() const {
    return { Monomial(std::make_shared<Sine>(*this)) };
}

Polynomial Power::toPolynomial() const {
    return { Monomial({*this}) };
}

Polynomial Monomial::toPolynomial() const {
    return { *this };
}