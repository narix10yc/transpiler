#include "qch/cas.h"

using namespace qch::cas;

Polynomial Constant::toPolynomial() const {
    return { std::make_shared<Monomial>(value) };
}

Polynomial Variable::toPolynomial() const {
    auto power = std::make_shared<Power>(std::make_shared<Variable>(*this));
    return { std::make_shared<Monomial>(power) };
}

Polynomial Cosine::toPolynomial() const {
    auto power = std::make_shared<Power>(std::make_shared<Cosine>(*this));
    return { std::make_shared<Monomial>(power) };
}

Polynomial Sine::toPolynomial() const {
    auto power = std::make_shared<Power>(std::make_shared<Sine>(*this));
    return { std::make_shared<Monomial>(power) };
}

Polynomial Power::toPolynomial() const {
    auto power = std::make_shared<Power>(*this);
    return { std::make_shared<Monomial>(power) };
}

Polynomial Monomial::toPolynomial() const {
    return { std::make_shared<Monomial>(*this) };
}