#ifndef QUENCH_POLYNOMIAL_H
#define QUENCH_POLYNOMIAL_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>

namespace quench::cas {

class Polynomial;

class CASNode {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual std::ostream& print(std::ostream&) const = 0;

    virtual std::ostream& printLaTeX(std::ostream&) const = 0;

    virtual expr_value getExprValue() const = 0;

    virtual bool equals(const CASNode*) const = 0;

    bool equals(std::shared_ptr<CASNode> p) const {
        return equals(p.get());
    }

    virtual int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() const = 0;

    virtual ~CASNode() = default;
};

class BasicCASNode : public CASNode {
public:
    virtual int compare(const BasicCASNode* other) const = 0;
};

class ConstantNode : public BasicCASNode {
    double value;
public:
    ConstantNode(double value) : value(value) {}
    
    std::ostream& print(std::ostream& os) const override {
        return os << value;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << value;
    }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherConstantNode = dynamic_cast<const ConstantNode*>(other);
        assert(otherConstantNode != nullptr);
        if (value < otherConstantNode->value)
            return -1;
        if (value == otherConstantNode->value)
            return 0;
        return +1;
    }

    bool equals(const CASNode* other) const override {
        if (auto otherConstantNode = dynamic_cast<const ConstantNode*>(other))
            return (otherConstantNode->value == value);
        return false;
    }

    int getSortPriority() const override { return 0; }

    Polynomial toPolynomial() const override;
};

class VariableNode : public BasicCASNode {
    std::string name;
public:
    VariableNode(const std::string& name) : name(name) {}

    std::ostream& print(std::ostream& os) const override {
        return os << name;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << name;
    }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherVariableNode = dynamic_cast<const VariableNode*>(other);
        assert(otherVariableNode != nullptr);

        return name.compare(otherVariableNode->name);
    }

    bool equals(const CASNode* other) const override {
        if (auto otherVariableNode = dynamic_cast<const VariableNode*>(other))
            return (otherVariableNode->name == name);
        return false;
    }

    int getSortPriority() const override { return 10; }

    Polynomial toPolynomial() const override;
};

class CosineNode : public BasicCASNode {
    std::shared_ptr<BasicCASNode> node;
public:
    CosineNode(std::shared_ptr<BasicCASNode> node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "cos(";
        node->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "\\cos(";
        node->print(os);
        os << ")";
        return os;
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::cos(nodeValue.value) };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        assert(otherCosineNode != nullptr);
        return node->compare(otherCosineNode->node.get());
    }

    bool equals(const CASNode* other) const override {
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        if (otherCosineNode == nullptr)
            return false;
        return (node->equals(otherCosineNode->node.get()));
    }

    int getSortPriority() const override { return 20; }

    Polynomial toPolynomial() const override;

};

class SineNode : public BasicCASNode {
    std::shared_ptr<BasicCASNode> node;
public:
    SineNode(std::shared_ptr<BasicCASNode> node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "sin(";
        node->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "\\sin(";
        node->print(os);
        os << ")";
        return os;
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        assert(otherSineNode != nullptr);
        return node->compare(otherSineNode->node.get());
    }

    bool equals(const CASNode* other) const override {
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        if (otherSineNode == nullptr)
            return false;
        return (node->equals(otherSineNode->node.get()));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    int getSortPriority() const override { return 30; }

    Polynomial toPolynomial() const override;
};

class Polynomial : public CASNode {
public:
    struct monomial_t {
        struct power_t {
            std::shared_ptr<BasicCASNode> base;
            int exponent = 1;
        };
        double coef = 1.0;
        std::vector<power_t> powers = {};

        int order() const {
            int sum = 0;
            for (const auto& p : powers)
                sum += p.exponent;
            return sum;
        }
    };

private:
    /// @brief monomial comparison function, strict order. return a < b 
    /// @return a < b. Happens when (1). a has less terms than b does, or 
    /// otherwise, (2). the order of a is less than the order of b, or (3) 
    static bool monomial_cmp(const monomial_t& a, const monomial_t& b) {
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

    static bool monomial_eq(const monomial_t& a, const monomial_t& b) {
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

    std::vector<monomial_t> monomials;

    Polynomial& operator+=(const monomial_t& monomial);
    
    Polynomial& operator-=(const monomial_t& monomial);

    Polynomial& operator*=(const monomial_t& monomial);
    
    void insertMonomial(const monomial_t& monomial) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
        monomials.insert(it, monomial);
    }
public:
    Polynomial() : monomials() {}
    Polynomial(double v) : monomials({{v, {}}}) {}
    Polynomial(std::initializer_list<monomial_t> monomials)
        : monomials(monomials) {}

    std::ostream& print(std::ostream& os) const override;

    std::ostream& printLaTeX(std::ostream& os) const override;

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly) {
        return poly.print(os);
    }

    bool equals(const CASNode* other) const override {
        assert(false && "Unimplemented yet");
        return false;
    }

    expr_value getExprValue() const override {
        double v = 0.0;
        double mV = 1.0;
        for (const auto& m : monomials) {
            mV = m.coef;
            for (const auto& p : m.powers) {
                auto baseV = p.base->getExprValue();
                if (!baseV.isConstant)
                    return { false };
                mV *= std::pow(baseV.value, p.exponent);
            }
            v += mV;
        }
        return { true, v };
    }

    int getSortPriority() const override { return 60; }

    Polynomial toPolynomial() const override { return Polynomial(*this); };

    Polynomial& operator+=(const Polynomial& other);

    Polynomial operator+(const Polynomial& other) const {
        // TODO: a better method
        Polynomial newPoly(*this);
        return newPoly += other;
    }

    Polynomial& operator-=(const Polynomial& other);

    Polynomial operator-(const Polynomial& other) const {
        // TODO: a better method
        Polynomial newPoly(*this);
        return newPoly -= other;
    }

    Polynomial& operator*=(const Polynomial& other) {
        auto newPoly = (*this) * other;
        return (*this) = newPoly;
    }

    Polynomial operator*(const Polynomial& other) const;
};

} // namespace quench::cas

#endif // QUENCH_POLYNOMIAL_H