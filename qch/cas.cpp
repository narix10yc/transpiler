#include "qch/cas.h"

using namespace qch::cas;

Polynomial* Constant::canonicalize(CASContext& ctx) const {
    auto monomial = ctx.get<Monomial>(value);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Variable::canonicalize(CASContext& ctx) const {
    auto variable = ctx.get<Variable>(name);
    auto power = ctx.get<Power>(variable);
    auto monomial = ctx.get<Monomial>(1.0, power);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Cosine::canonicalize(CASContext& ctx) const {
    auto cosine = ctx.get<Cosine>(node);
    auto power = ctx.get<Power>(cosine);
    auto monomial = ctx.get<Monomial>(1.0, power);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Sine::canonicalize(CASContext& ctx) const {
    auto sine = ctx.get<Sine>(node);
    auto power = ctx.get<Power>(sine);
    auto monomial = ctx.get<Monomial>(1.0, power);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Power::canonicalize(CASContext& ctx) const {
    auto power = ctx.get<Power>(base, exponent);
    auto monomial = ctx.get<Monomial>(1.0, power);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Monomial::canonicalize(CASContext& ctx) const {
    auto monomial = ctx.get<Monomial>(coefficient, pows);
    auto polynomial = ctx.get<Polynomial>(monomial);
    return polynomial;
}

Polynomial* Polynomial::canonicalize(CASContext& ctx) const {
    auto p = ctx.get<Polynomial>();
    for (auto m : monomials)
        p->addInplace(m);
    return p;
}

void Polynomial::scalarMulInPlace(double scalar) {
    for (auto m : monomials)
        m->coefficient *= scalar;
}

void Polynomial::addInplace(Monomial* other) {
    for (auto monomial : monomials) {
        if (monomial->equals(other)) {
            monomial->coefficient += other->coefficient;
            return;
        }
    } 
    monomials.insert(other);
}

void Polynomial::addInplace(Polynomial* other) {
    for (auto* monomial : other->monomials)
        addInplace(monomial);
}