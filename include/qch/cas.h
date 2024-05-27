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

    /// @brief Compare with another node
    /// @return -1: less than; 0: equal; +1: greater than
    virtual int compare(const Node*) const = 0;

    virtual bool equals(const Node*) const = 0;

    bool equals(std::shared_ptr<Node> p) const {
        return equals(p.get());
    }

    virtual inline int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() const = 0;
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

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherPower = dynamic_cast<const Power*>(other);
        assert(otherPower != nullptr);
        return base->compare(otherPower->base.get());
    }

    bool equals(const Node* other) const override {
        auto otherPower = dynamic_cast<const Power*>(other);
        if (otherPower == nullptr)
            return false;
        if (otherPower->exponent != exponent)
            return false;
        return (base->equals(otherPower->base.get()));
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
    std::vector<std::shared_ptr<Power>> pows;

    Monomial(double coef) : coefficient(coef), pows() {}
    Monomial(std::shared_ptr<Power> pow, double coef = 1.0)
        : coefficient(coef), pows({pow}) {}

    int order() const {
        int s = 0;
        for (const auto pow : pows)
            s += pow->exponent;
        return s;
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
            pows[i]->print(os);
            os << " * ";
        }
        pows[length-1]->print(os);
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherMonomial = dynamic_cast<const Monomial*>(other);
        assert(otherMonomial != nullptr);
        if (pows.size() < otherMonomial->pows.size())
            return -1;
        if (pows.size() > otherMonomial->pows.size())
            return +1; 

        for (unsigned i = 0; i < pows.size(); i++) {
            int r = pows[i]->compare(otherMonomial->pows[i].get());
            if (r == -1) return -1;
            if (r == +1) return +1;
        }
        return 0;
    }

    bool equals(const Node* other) const override {
        auto otherMonomial = dynamic_cast<const Monomial*>(other);
        if (otherMonomial == nullptr)
            return false;
        if (otherMonomial->coefficient != coefficient)
            return false;
        if (otherMonomial->pows.size() != pows.size())
            return false;

        for (unsigned i = 0; i < pows.size(); i++) {
            if (!(pows[i]->equals(otherMonomial->pows[i].get())))
                return false;
        }
        return true;
    }

    expr_value getExprValue() const override {
        expr_value value { true, coefficient };
        expr_value powValue;
        for (const auto pow : pows) {
            powValue = pow->getExprValue();
            if (!powValue.isConstant)
                return { false };
            value.value *= powValue.value;
        }
        return value;
    }

    inline int getSortPriority() const override { return 50; }

    Monomial& operator*= (const Power& other) {
        for (unsigned i = 0; i < pows.size(); i++) {
            if (pows[i]->base->equals(other.base.get())) {
                auto newPaw = std::make_shared<Power>(*pows[i]);
                newPaw->exponent += other.exponent;
                pows[i] = newPaw;
                return *this;
            }
        }
        pows.push_back(std::make_shared<Power>(other));
        return *this;
    }

    std::shared_ptr<Monomial> tryAddWith(const Monomial* other) {
        if (pows.size() != other->pows.size())
            return nullptr;
        for (unsigned i = 0; i < pows.size(); i++) {
            if (!(pows[i]->equals(other->pows[i].get())))
                return nullptr;
        }
        auto pMonomial = std::make_shared<Monomial>(*this);
        pMonomial->coefficient += other->coefficient;
        return pMonomial;
    }

    Monomial operator* (const Power& other) const {
        auto newMonomial(*this);
        return newMonomial *= other;
    }

    Polynomial toPolynomial() const override;

};


class Polynomial : public Node {
public:
    std::vector<std::shared_ptr<Monomial>> monomials;
    
    Polynomial() : monomials() {}
    Polynomial(std::shared_ptr<Monomial> m) : monomials({m}) {}

    void sort() {
        std::sort(monomials.begin(), monomials.end());
    }

    void print(std::ostream& os) const override {
        auto size = monomials.size();
        if (size == 0)
            return;

        for (unsigned i = 0; i < size-1; i++) {
            monomials[i]->print(os);
            os << " + ";
        }
        monomials[size-1]->print(os);
    }

    int compare(const Node* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherPolynomial = dynamic_cast<const Polynomial*>(other);
        assert(otherPolynomial != nullptr);
        if (monomials.size() < otherPolynomial->monomials.size())
            return -1;
        if (monomials.size() < otherPolynomial->monomials.size())
            return -1;

        for (unsigned i = 0; i < monomials.size(); i++) {
            int r = monomials[i]->compare(otherPolynomial->monomials[i].get());
            if (r == -1) return -1;
            if (r == +1) return +1;
        }
        return 0;
    }

    bool equals(const Node* other) const override {
        auto otherPolynomial = dynamic_cast<const Polynomial*>(other);
        if (otherPolynomial == nullptr)
            return false;
        if (otherPolynomial->monomials.size() != monomials.size())
            return false;
        
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (!(monomials[i]->equals(otherPolynomial->monomials[i].get())))
                return false;
        }
        return true;
    }

    expr_value getExprValue() const override {
        expr_value value { true, 0.0 };
        expr_value monomialValue;

        for (const auto monomial : monomials) {
            monomialValue = monomial->getExprValue();
            if (!monomialValue.isConstant)
                return { false };
            value.value += monomialValue.value;
        }
        return value;
    }

    inline int getSortPriority() const override { return 60; }

    Polynomial toPolynomial() const override { return *this; }

    void addMonomialInPlace(std::shared_ptr<Monomial> m) {
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (auto pMonomial = monomials[i]->tryAddWith(m.get())) {
                monomials[i] = pMonomial;
                return;
            }
        }
        monomials.push_back(m);
    }

    Polynomial operator+ (const Polynomial& other) const {
        Polynomial newPoly(*this);
        for (const auto monomial : other.monomials)
            newPoly.addMonomialInPlace(monomial);
        return newPoly;
    }
};



} // namespace qch::cas


#endif // QCH_CAS_H