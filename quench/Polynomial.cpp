#include "quench/Polynomial.h"
#include "utils/iocolor.h"

using namespace quench::cas;
using namespace Color;

bool Polynomial::monomial_cmp(const monomial_t& a, const monomial_t& b) {
    auto aSize = a.powers.size();
    auto bSize = b.powers.size();
    if (aSize < bSize) return true;
    if (aSize > bSize) return false;
    auto aOrder = a.order();
    auto bOrder = b.order();
    if (aOrder < bOrder) return true;
    if (aOrder > bOrder) return false;
    for (unsigned i = 0; i < aSize; i++) {
        int r = a.powers[i].base->compare(b.powers[i].base.get());
        if (r < 0) return true;
        if (r > 0) return false;
        if (a.powers[i].exponent > b.powers[i].exponent)
            return true;
        if (a.powers[i].exponent < b.powers[i].exponent)
            return false;
    }
    return false;
};

bool Polynomial::monomial_eq(const monomial_t& a, const monomial_t& b) {
    auto aSize = a.powers.size();
    auto bSize = b.powers.size();
    if (aSize != bSize)
        return false;
    if (a.order() != b.order())
        return false;
    for (unsigned i = 0; i < aSize; i++) {
        if (a.powers[i].exponent != b.powers[i].exponent)
            return false;
        if (!(a.powers[i].base->equals(b.powers[i].base.get())))
            return false;
    }
    return true;
}

std::ostream& Polynomial::print(std::ostream& os) const {
    if (monomials.empty())
        return os << "0";
    if (monomials.size() == 1 && monomials[0].powers.empty())
        return os << monomials[0].coef;

    auto it = monomials.begin();
    while (it != monomials.end()) {
        if (it != monomials.begin())
            os << " + ";
        if (it->coef == 1.0) {}
        else if (it->coef == -1.0) { os << "-"; }
        else { os << it->coef; }
        for (const auto& p : it->powers) {
            p.base->print(os);
            if (p.exponent != 1)
                os << "**" << p.exponent << "";
        }
        it++;
    }
    return os;
}

std::ostream& Polynomial::printLaTeX(std::ostream& os) const {
    if (monomials.empty())
        return os << "0";
    if (monomials.size() == 1 && monomials[0].powers.empty())
        return os << monomials[0].coef;

    auto it = monomials.begin();
    while (it != monomials.end()) {
        if (it != monomials.begin())
            os << " + ";
        if (it->coef == 1.0) {}
        else if (it->coef == -1.0) { os << "-"; }
        else { os << it->coef << " "; }
        for (const auto& p : it->powers) {
            p.base->printLaTeX(os);
            if (p.exponent != 1)
                os << "^{" << p.exponent << "}";
        }
        it++;
    }
    return os;
}

Polynomial& Polynomial::operator+=(const monomial_t& monomial) {
    auto it = std::lower_bound(monomials.begin(), monomials.end(),
                               monomial, monomial_cmp);
    if (it == monomials.end())
        monomials.insert(it, monomial);
    else if (monomial_eq(*it, monomial))
        it->coef += monomial.coef;
    else
        monomials.insert(it, monomial);
    return *this;
}

Polynomial& Polynomial::operator-=(const monomial_t& monomial) {
    auto it = std::lower_bound(monomials.begin(), monomials.end(),
                               monomial, monomial_cmp);
    if (it == monomials.end())
        monomials.insert(it, monomial)->coef *= -1.0;
    else if (monomial_eq(*it, monomial))
        it->coef -= monomial.coef;
    else
        monomials.insert(it, monomial)->coef *= -1.0;
    return *this;
}

Polynomial& Polynomial::operator+=(const Polynomial& other) {
    for (const auto& monomial : other.monomials)
        operator+=(monomial);
    return *this;
}

Polynomial& Polynomial::operator-=(const Polynomial& other) {
    for (const auto& monomial : other.monomials)
        operator-=(monomial);
    return *this;
}

Polynomial& Polynomial::operator*=(const monomial_t& m) {
    for (auto& monomial : monomials) {
        monomial.coef *= m.coef;
        auto myIter = monomial.powers.begin();
        auto otherIter = m.powers.begin();
        while (true) {
            if (otherIter == m.powers.end())
                break;
            if (myIter == monomial.powers.end()) {
                // std::cerr << "flag2\n";
                while (otherIter != m.powers.end()) {
                    monomial.powers.push_back(*otherIter);
                    otherIter++;
                }
                break;
            }
            int cmp = myIter->base->compare(otherIter->base.get());
            if (cmp == 0) {
                myIter->exponent += otherIter->exponent;
                myIter++; otherIter++;
            }
            else if (cmp < 0)
                myIter++;
            else {
                monomial.powers.insert(myIter, *otherIter);
                otherIter++;
            }
        } // while loop
    }
    return *this;
}

Polynomial Polynomial::operator*(const Polynomial& other) const {
    Polynomial newPoly;
    for (const auto& m : other.monomials) {
        auto tmp = *this;
        tmp *= m;
        newPoly += tmp;
    }
    return newPoly;
}

Polynomial ConstantNode::toPolynomial() const {
    return { value };
}

Polynomial VariableNode::toPolynomial() const {
    return {{1.0, {{std::make_shared<VariableNode>(*this), 1}}}};
}

Polynomial CosineNode::toPolynomial() const {
    return {{1.0, {{std::make_shared<CosineNode>(*this), 1}}}};
}

Polynomial SineNode::toPolynomial() const {
    return {{1.0, {{std::make_shared<SineNode>(*this), 1}}}};
}

Polynomial VarAddNode::toPolynomial() const {
    return {{1.0, {{std::make_shared<VarAddNode>(*this), 1}}}};
}
Polynomial ComplexExpNode::toPolynomial() const {
    return {{1.0, {{std::make_shared<ComplexExpNode>(*this), 1}}}};
}