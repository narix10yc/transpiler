#include "qch/GateMatrix.h"

using namespace qch::ir;

std::ostream& Polynomial::print(std::ostream& os) const {
    if (monomials.empty())
        return os;

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
        return os;

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


// using monomial_t = Polynomial::monomial_t;
// using power_t = monomial_t::power_t;

Polynomial& Polynomial::operator+=(const monomial_t& monomial) {
    // std::cerr << "about to add monomial " << monomial << " to ";
    // print(std::cerr) << "\n";
    auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
    if (it == monomials.end()) {
        // std::cerr << "iter is end, just insert it\n";
        monomials.insert(it, monomial);
    }
    else if (monomial_eq(*it, monomial)) {
        // std::cerr << "Lower bound element is the same, add coef together\n";
        it->coef += monomial.coef;
    }
    else {
        // std::cerr << "The lower bound element " << (*it) << " is different, "
                    //  "insert monomial " << monomial << "\n";
        monomials.insert(it, monomial);
    }
    // std::cerr << "addition result = ";
    // print(std::cerr) << "\n";
    return *this;
}

Polynomial& Polynomial::operator+=(const Polynomial& other) {
    for (const auto& monomial : other.monomials)
        operator+=(monomial);

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
                // std::cerr << "flag3\n";
                myIter->exponent += otherIter->exponent;
                myIter++; otherIter++;
            }
            else if (cmp < 0) {
                                // std::cerr << "flag4\n";
                myIter++;
            }
            else {
                                // std::cerr << "flag4\n";
                monomial.powers.insert(myIter, *otherIter);
                otherIter++;
            }
        } // while loop
    }
    return *this;
}


Polynomial Polynomial::operator*(const Polynomial& other) {
    std::cerr << "Start to multiply ";
    print(std::cerr) << " with ";
    other.print(std::cerr) << "\n";
    Polynomial newPoly;
    for (const auto& m : other.monomials) {
        auto tmp = *this;
        std::cerr << "sub step: mul with m = " << m << "\n tmp = ";
        (tmp *= m).print(std::cerr) << "\n";
        newPoly += tmp;
        std::cerr << "newPoly is now ";
        newPoly.print(std::cerr) << "\n";
    }
    std::cerr << "result: ";
    newPoly.print(std::cerr) << "\n";
    return newPoly;
}


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