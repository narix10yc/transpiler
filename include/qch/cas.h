#ifndef QCH_CAS_H
#define QCH_CAS_H

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include <utility>
#include <memory>
#include <cmath>

namespace qch::cas {

class Node;
class Polynomial;
class CASContext;


class Node {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual void print(std::ostream&) const = 0;

    virtual expr_value getExprValue() const = 0;

    /// @brief Compare with another node. In some types of Nodes, compare == 0
    /// does 'not' mean equal. This includes:
    /// exponent of the Power class
    /// coefficient of the Monomial class
    /// @return -1: less than; 0: equal; +1: greater than
    virtual int compare(const Node*) const = 0;

    virtual bool equals(const Node*) const = 0;

    bool equals(std::shared_ptr<Node> p) const {
        return equals(p.get());
    }

    virtual inline int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() const = 0;

    ~Node() {
        // std::cerr << "destructor\n";
    }
};

class BasicNode : public Node {

};


class Constant : public BasicNode {
    double value;
public:
    Constant(double value) : value(value) {}
    
    void print(std::ostream& os) const override { os << value; }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherConstant = dynamic_cast<const Constant*>(other);
        assert(otherConstant != nullptr);
        if (value < otherConstant->value)
            return -1;
        if (value == otherConstant->value)
            return 0;
        return +1;
    }

    bool equals(const Node* other) const override {
        if (auto otherConstant = dynamic_cast<const Constant*>(other))
            return (otherConstant->value == value);
        return false;
    }

    inline int getSortPriority() const override { return 0; }

    Polynomial toPolynomial() const override;
};


class Variable : public BasicNode {
    std::string name;
public:
    Variable(const std::string& name) : name(name) {}

    void print(std::ostream& os) const override { os << name; }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherVariable = dynamic_cast<const Variable*>(other);
        assert(otherVariable != nullptr);

        return name.compare(otherVariable->name);
    }

    bool equals(const Node* other) const override {
        if (auto otherVariable = dynamic_cast<const Variable*>(other))
            return (otherVariable->name == name);
        return false;
    }

    inline int getSortPriority() const override { return 10; }

    Polynomial toPolynomial() const override;
};


class Cosine : public BasicNode {
    std::shared_ptr<BasicNode> node;
public:
    Cosine(std::shared_ptr<BasicNode> node) : node(node) {}

    void print(std::ostream& os) const override {
        os << "cos(";
        node->print(os);
        os << ")";
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::cos(nodeValue.value) };
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosine = dynamic_cast<const Cosine*>(other);
        assert(otherCosine != nullptr);
        return node->compare(otherCosine->node.get());
    }

    bool equals(const Node* other) const override {
        auto otherCosine = dynamic_cast<const Cosine*>(other);
        if (otherCosine == nullptr)
            return false;
        return (node->equals(otherCosine->node.get()));
    }

    inline int getSortPriority() const override { return 20; }

    Polynomial toPolynomial() const override;

};


class Sine : public BasicNode {
    std::shared_ptr<BasicNode> node;
public:
    Sine(std::shared_ptr<BasicNode> node) : node(node) {}

    void print(std::ostream& os) const override {
        os << "sin(";
        node->print(os);
        os << ")";
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSine = dynamic_cast<const Sine*>(other);
        assert(otherSine != nullptr);
        return node->compare(otherSine->node.get());
    }

    bool equals(const Node* other) const override {
        auto otherSine = dynamic_cast<const Sine*>(other);
        if (otherSine == nullptr)
            return false;
        return (node->equals(otherSine->node.get()));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    inline int getSortPriority() const override { return 30; }

    Polynomial toPolynomial() const override;

};


class Power : public Node {
public:
    std::shared_ptr<BasicNode> base;
    int exponent;
    Power(std::shared_ptr<BasicNode> base, int exponent = 1)
        : base(base), exponent(exponent) {}

    void print(std::ostream& os) const override {
        base->print(os);
        if (exponent != 1.0)
            os << "**" << exponent;
    }

    int compare(const Power& other) const {
        return base->compare(other.base.get());
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherPower = dynamic_cast<const Power*>(other);
        assert(otherPower != nullptr);
        return compare(*otherPower);
    }

    bool operator==(const Power& other) const {
        if (exponent != other.exponent)
            return false;
        return base->equals(other.base.get());
    }

    bool operator!=(const Power& other) const {
        if (exponent != other.exponent)
            return true;
        return !(base->equals(other.base.get()));
    }

    bool equals(const Node* other) const override {
        auto otherPower = dynamic_cast<const Power*>(other);
        if (otherPower == nullptr)
            return false;
        return (*this) == (*otherPower);
    }

    expr_value getExprValue() const override {
        auto baseValue = base->getExprValue();
        if (!baseValue.isConstant)
            return { false };
        return { true, std::pow(baseValue.value, exponent) };
    }

    inline int getSortPriority() const override { return 40; }

    Polynomial toPolynomial() const override;

};

class Monomial : public Node {
public:
    double coefficient;
    std::vector<Power> pows;

    Monomial(double coef) : coefficient(coef), pows() {}
    Monomial(std::shared_ptr<BasicNode> basicNode, double coef = 1.0) 
        : coefficient(coef), pows({Power{basicNode}}) {}
    Monomial(std::initializer_list<Power> pows, double coef = 1.0)
        : coefficient(coef), pows(pows) {}

    int degree() const {
        int s = 0;
        for (const auto& pow : pows)
            s += pow.exponent;
        return s;
    }

    void sortSelf() {
        std::sort(pows.begin(), pows.end(),
            [](const Power& a, const Power& b) { return a.compare(b) < 0; });
    }

    Monomial sortAndSimplify() const {
        Monomial newMonomial(coefficient);
        // this makes sure no replicates
        for (const auto& pow : pows)
            newMonomial *= pow;
        newMonomial.sortSelf();
        return newMonomial;
    }

    void print(std::ostream& os) const override {
        auto length = pows.size();
        if (length == 0) {
            os << coefficient;
            return;
        }

        if (coefficient != 1.0)
            os << coefficient;
        
        for (unsigned i = 0; i < length-1; i++) {
            pows[i].print(os);
            os << " ";
        }
        pows[length-1].print(os);
    }

    int compare(const Monomial& other) const {
        if (degree() < other.degree())
            return -1;
        if (degree() > other.degree())
            return +1;
        if (pows.size() < other.pows.size())
            return -1;
        if (pows.size() > other.pows.size())
            return +1;

        auto thisIt = pows.begin();
        auto otherIt = other.pows.begin();
        while (thisIt != pows.end()) {
            int r = (*thisIt).compare(*otherIt);
            if (r == -1) return -1;
            if (r == +1) return +1;
            ++thisIt; ++otherIt;
        }
        return 0;
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherMonomial = dynamic_cast<const Monomial*>(other);
        assert(otherMonomial != nullptr);
        return compare(*otherMonomial);
    }

    bool equals(const Node* other) const override {
        auto otherMonomial = dynamic_cast<const Monomial*>(other);
        if (otherMonomial == nullptr)
            return false;
        if (otherMonomial->coefficient != coefficient)
            return false;
        if (otherMonomial->pows.size() != pows.size())
            return false;

        auto thisIt = pows.begin();
        auto otherIt = otherMonomial->pows.begin();
        while (thisIt != pows.end()) {
            if ((*thisIt) != (*otherIt))
                return false;
            ++thisIt; ++otherIt;
        }
        return true;
    }

    expr_value getExprValue() const override {
        expr_value value { true, coefficient };
        expr_value powValue;
        for (const auto& pow : pows) {
            powValue = pow.getExprValue();
            if (!powValue.isConstant)
                return { false };
            value.value *= powValue.value;
        }
        return value;
    }

    inline int getSortPriority() const override { return 50; }

    bool operator==(const Monomial& other) const {
        if (coefficient != other.coefficient)
            return false;
    
        auto thisIt = pows.begin();
        auto otherIt = other.pows.begin();
        while (thisIt != pows.end()) {
            if ((*thisIt) != (*otherIt))
                return false;
            ++thisIt; ++otherIt;
        }
        return true;
    }

    bool operator!=(const Monomial& other) const {
        if (coefficient != other.coefficient)
            return true;

        auto thisIt = pows.begin();
        auto otherIt = other.pows.begin();
        while (thisIt != pows.end()) {
            if ((*thisIt) != (*otherIt))
                return true;
            ++thisIt; ++otherIt;
        }
        return false;
    }   

    Monomial& operator*=(const Power& p) {
        for (auto& pow : pows) {
            if (pow.compare(p) == 0) {
                pow.exponent += p.exponent;
                return *this;
            }
        }
        pows.push_back(p);
        return *this;
    }

    Monomial operator*(const Power& other) const {
        Monomial newMonomial(*this);
        return newMonomial *= other;
    }

    Monomial& operator*=(const Monomial& other) {
        for (const auto& pow : other.pows)
            (*this) *= pow;
        this->coefficient *= other.coefficient;
        return *this;
    }

    Monomial operator*(const Monomial& other) const {
        Monomial newMonomial(*this);
        return newMonomial *= other;
    }

    Polynomial toPolynomial() const override;
};


class Polynomial : public Node {
    struct monomial_cmp_less {
        bool operator()(const Monomial& a, const Monomial& b) const {
            return a.compare(b) < 0;
        }
    };
public:
    std::vector<Monomial> monomials;
    
    Polynomial() : monomials() {}
    Polynomial(std::initializer_list<Monomial> monomials)
        : monomials(monomials) {}

    void sortSelf() {
        std::sort(monomials.begin(), monomials.end(),
            [](const Monomial& a, const Monomial& b) {
                return a.compare(b) < 0;
            });
    }

    Polynomial sortAndSimplify() const {
        Polynomial newPoly;
        // this makes sure no replicates
        for (const auto& monomial : monomials)
            newPoly += monomial;
        newPoly.sortSelf();
        return newPoly;
    }

    void print(std::ostream& os) const override {
        auto size = monomials.size();
        if (size == 0)
            return;

        for (unsigned i = 0; i < size-1; i++) {
            monomials[i].print(os);
            os << " + ";
        }
        monomials[size-1].print(os);
    }

    int compare(const Polynomial& other) const {
        if (monomials.size() < other.monomials.size())
            return -1;
        if (monomials.size() < other.monomials.size())
            return +1;

        for (unsigned i = 0; i < monomials.size(); i++) {
            int r = monomials[i].compare(other.monomials[i]);
            if (r == -1) return -1;
            if (r == +1) return +1;
        }
        return 0;
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherPolynomial = dynamic_cast<const Polynomial*>(other);
        assert(otherPolynomial != nullptr);
        return compare(*otherPolynomial);
    }

    bool operator==(const Polynomial& other) const {
        if (monomials.size() != other.monomials.size())
            return false;
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i] != other.monomials[i])
                return false;
        }
        return true;
    }

    bool operator!=(const Polynomial& other) const {
        if (monomials.size() != other.monomials.size())
            return true;
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i] != other.monomials[i])
                return true;
        }
        return false;
    }

    bool equals(const Node* other) const override {
        auto otherPolynomial = dynamic_cast<const Polynomial*>(other);
        if (otherPolynomial == nullptr)
            return false;
        return (*this) == (*otherPolynomial);
    }

    expr_value getExprValue() const override {
        expr_value value { true, 0.0 };
        expr_value monomialValue;

        for (const auto& monomial : monomials) {
            monomialValue = monomial.getExprValue();
            if (!monomialValue.isConstant)
                return { false };
            value.value += monomialValue.value;
        }
        return value;
    }

    Polynomial& operator+=(const Monomial& m) {
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i].compare(m) == 0) {
                monomials[i].coefficient += m.coefficient;
                return *this;
            }
        }
        monomials.push_back(m);
        return *this;
    }

    Polynomial& operator+=(const Polynomial& other) {
        for (const auto& monomial : other.monomials)
            (*this) += monomial;
        return *this;
    }

    Polynomial operator+(const Polynomial& other) const {
        Polynomial newPoly(*this);
        return newPoly += other;
    }

    Polynomial& operator*=(const Monomial& m) {
        for (auto& monomial : monomials)
            monomial *= m;
        return *this;
    }

    Polynomial operator*(const Monomial& m) const {
        Polynomial newPoly(*this);
        return newPoly *= m;
    }

    Polynomial& operator*=(const Polynomial& other) {
        for (const auto& m : other.monomials)
            (*this) *= m;
        return *this;
    }

    Polynomial operator*(const Polynomial& other) const {
        Polynomial newPoly(*this);
        return newPoly *= other;
    }

    inline int getSortPriority() const override { return 60; }

    Polynomial toPolynomial() const override { return *this; }
};



} // namespace qch::cas


#endif // QCH_CAS_H