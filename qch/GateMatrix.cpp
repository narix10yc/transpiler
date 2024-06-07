#include "qch/GateMatrix.h"

using namespace qch::ir;

std::ostream& Polynomial::print(std::ostream& os) const {
    if (monomials.empty())
        return os;

    for (unsigned i = 0; i < monomials.size()-1; i++)
        os << monomials[i] << " + ";
    return os << monomials.back();
}

Polynomial& Polynomial::operator+=(const Polynomial& other) {
    for (const auto& monomial : other.monomials) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
        std::cerr << "about to add " << monomial << "\n";
        if (it == monomials.end()) {
            std::cerr << "iter is end, just insert it\n";
            monomials.insert(it, monomial);
            continue;
        }
        if (monomial_cmp(monomial, *it)) {
            std::cerr << "The lower bound element " << (*it) << " is different \n";
            monomials.insert(it, monomial);
            continue;
        }
        std::cerr << "Lower bound element is the same\n";
        it->coef += monomial.coef;
    }

    std::cerr << "addition finished ";
    print(std::cerr) << "\n";
    return *this;
}


using monomial_t = Polynomial::monomial_t;
using power_t = monomial_t::power_t;

Polynomial ConstantNode::toPolynomial() const {
    return {{value, {}}};
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