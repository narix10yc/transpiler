#ifndef QCH_GATE_MATRIX_H
#define QCH_GATE_MATRIX_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

namespace qch::ir {

class Polynomial;

class CASNode {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual std::ostream& print(std::ostream&) const = 0;

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
    };

    friend std::ostream& operator<<(std::ostream& os, const monomial_t::power_t& power) {
        power.base->print(os);
        if (power.exponent != 1)
            os << "**" << power.exponent;
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const monomial_t& m) {
        if (m.coef == 1.0) {}
        else if (m.coef == -1.0) { os << "-"; }
        else { os << m.coef; }
        for (const auto& p : m.powers) {
            p.base->print(os);
            if (p.exponent != 1)
                os << "**" << p.exponent;
        }
        return os;
    }

private:
    static inline bool monomial_cmp(const monomial_t& a, const monomial_t& b) {
        auto aSize = a.powers.size();
        auto bSize = b.powers.size();
        if (aSize < bSize) return true;
        if (aSize > bSize) return false;
        for (unsigned i = 0; i < aSize; i++) {
            int r = a.powers[i].base->compare(b.powers[i].base.get());
            if (r < 0) return true;
            if (r > 0) return false;
        }
        return false;
    };

    std::vector<monomial_t> monomials;
    
    void insertMonomial(const monomial_t& monomial) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
        monomials.insert(it, monomial);
    }
public:
    Polynomial() : monomials() {}
    Polynomial(std::initializer_list<monomial_t> monomials)
        : monomials(monomials) {}

    std::ostream& print(std::ostream& os) const override;

    bool equals(const CASNode* other) const override {
        return false;
    }

    expr_value getExprValue() const override {
        double v = 0.0;
        double mV = 1.0;
        for (const auto& m : monomials) {
            mV = 1.0;
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

    Polynomial operator+(const Polynomial& other) {
        // TODO: a better method
        Polynomial newPoly(*this);
        return newPoly += other;
    }

    Polynomial& operator*=(const Polynomial& other);

    Polynomial operator*(const Polynomial& other) {
        Polynomial newPoly(*this);
        return newPoly *= other;
    }
};



} // namespace qch::ir

#endif // QCH_GATE_MATRIX_H