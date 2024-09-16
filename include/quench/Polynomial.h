#ifndef QUENCH_POLYNOMIAL_H
#define QUENCH_POLYNOMIAL_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>
#include <complex>

namespace quench::cas {

class Polynomial;

class CasNode {
public:
    struct expr_value {
        bool isConstant;
        std::complex<double> value;
    };

    virtual std::ostream& print(std::ostream&) const = 0;

    virtual expr_value getExprValue() const = 0;

    virtual int compare(const CasNode* other) const = 0;

    virtual bool equals(const CasNode* other) const {
        return compare(other) == 0;
    }

    virtual int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() = 0;

    virtual ~CasNode() = default;
};

class ConstantNode : public CasNode {
public:
    std::complex<double> value;

    ConstantNode(const std::complex<double>& value) : value(value) {}
    
    std::ostream& print(std::ostream& os) const override {
        return os << value;
    }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherConstantNode = dynamic_cast<const ConstantNode*>(other);
        assert(otherConstantNode != nullptr);
        if (value == otherConstantNode->value)
            return 0;
        if (value.real() < otherConstantNode->value.real())
            return -1;
        if (value.real() > otherConstantNode->value.real())
            return +1;
        if (value.imag() < otherConstantNode->value.imag())
            return -1;
        if (value.imag() > otherConstantNode->value.imag())
            return +1;
        assert(false && "Unreachable");
        return 0;
    }

    bool equals(const CasNode* other) const override {
        if (auto otherConstantNode = dynamic_cast<const ConstantNode*>(other))
            return (otherConstantNode->value == value);
        return false;
    }

    int getSortPriority() const override { return 0; }

    Polynomial toPolynomial() override;
};

class VariableNode : public CasNode {
public:
    int v;

    VariableNode(int v) : v(v) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "%" << v;
    }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto* otherVariableNode = dynamic_cast<const VariableNode*>(other);
        assert(otherVariableNode != nullptr);

        if (v > otherVariableNode->v)
            return +1;
        if (v == otherVariableNode->v)
            return 0;
        return -1;
    }

    bool equals(const CasNode* other) const override {
        if (auto otherVariableNode = dynamic_cast<const VariableNode*>(other))
            return (otherVariableNode->v == v);
        return false;
    }

    int getSortPriority() const override { return 10; }

    Polynomial toPolynomial() override;
};

class CosineNode : public CasNode {
    CasNode* node;
public:
    CosineNode(CasNode* node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "cos";
        return node->print(os);
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::cos(nodeValue.value) };
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        assert(otherCosineNode != nullptr);
        return node->compare(otherCosineNode->node);
    }

    bool equals(const CasNode* other) const override {
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        if (otherCosineNode == nullptr)
            return false;
        return (node->equals(otherCosineNode->node));
    }

    int getSortPriority() const override { return 20; }

    Polynomial toPolynomial() override;

};

class SineNode : public CasNode {
    CasNode* node;
public:
    SineNode(CasNode* node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "sin";
        return node->print(os);
        return os;
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        assert(otherSineNode != nullptr);
        return node->compare(otherSineNode->node);
    }

    bool equals(const CasNode* other) const override {
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        if (otherSineNode == nullptr)
            return false;
        return (node->equals(otherSineNode->node));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    int getSortPriority() const override { return 30; }

    Polynomial toPolynomial() override;
};

class AddNode : public CasNode {
    CasNode* lhs;
    CasNode* rhs;
public:
    AddNode(CasNode* lhs, CasNode* rhs) : lhs(lhs), rhs(rhs) {}

    std::ostream& print(std::ostream& os) const override {
        os << "(";
        lhs->print(os);
        os << "+";
        rhs->print(os);
        os << ")";
        return os;
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherAddNode = dynamic_cast<const AddNode*>(other);
        assert(otherAddNode != nullptr);

        auto compareLHS = lhs->compare(otherAddNode->lhs);
        if (compareLHS != 0)
            return compareLHS;
        return rhs->compare(otherAddNode->rhs);
    }

    bool equals(const CasNode* other) const override {
        auto otherAddNode = dynamic_cast<const AddNode*>(other);
        if (otherAddNode == nullptr)
            return false;
        return (lhs->equals(otherAddNode->lhs))  && (rhs->equals(otherAddNode->rhs));
    }

    expr_value getExprValue() const override {
        auto lhsValue = lhs->getExprValue();
        if (!lhsValue.isConstant)
            return { false };
        auto rhsValue = rhs->getExprValue();
        if (!rhsValue.isConstant)
            return { false };
        return { true, lhsValue.value + rhsValue.value };
    }

    int getSortPriority() const override { return 15; }

    Polynomial toPolynomial() override;
};

class ComplexExpNode : public CasNode {
    CasNode* node;
public:
    ComplexExpNode(CasNode* node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "cexp";
        node->print(os);
        os << "";
        return os;
    }

    int compare(const CasNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCExpNode = dynamic_cast<const ComplexExpNode*>(other);
        assert(otherCExpNode != nullptr);
        return node->compare(otherCExpNode->node);
    }

    bool equals(const CasNode* other) const override {
        if (auto otherCExpNode = dynamic_cast<const ComplexExpNode*>(other))
            return (node->equals(otherCExpNode->node));
        return false;
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        std::complex<double> ivalue(-nodeValue.value.imag(), nodeValue.value.real());
        return { true, std::exp(ivalue) };
    }

    int getSortPriority() const override { return 40; }

    Polynomial toPolynomial() override;
};

class Polynomial : public CasNode {
public:
    struct monomial_t {
        struct power_t {
            CasNode* base;
            int exponent;

            power_t(CasNode* base, int exponent=1) : base(base), exponent(exponent) {}
        };
        std::complex<double> coef;
        std::vector<power_t> powers;

        int order() const {
            int sum = 0;
            for (const auto& p : powers)
                sum += p.exponent;
            return sum;
        }

        monomial_t(const std::complex<double>& coef = {1.0, 0.0},
                   const std::vector<power_t>& powers = {})
            : coef(coef), powers(powers) {}
        
        monomial_t(CasNode* base) : coef(1.0, 0.0), powers({ {base, 1} }) {}
    };

private:
    /// @brief monomial comparison function (coef neglected), strict order
    /// @return a < b. Happens when (1). a has less terms than b does, or 
    /// otherwise, (2). the order of a is less than the order of b, or (3) 
    static bool monomial_cmp(const monomial_t& a, const monomial_t& b);
    static bool monomial_eq(const monomial_t& a, const monomial_t& b);

    std::vector<monomial_t> monomials;

    Polynomial& operator+=(const monomial_t& monomial);
    
    Polynomial& operator-=(const monomial_t& monomial);

    Polynomial& operator*=(const monomial_t& monomial);
public:
    Polynomial() : monomials() {}
    Polynomial(const monomial_t& m) : monomials({m}) {}
    Polynomial(const std::vector<monomial_t>& monomials) {
        for (const auto& m : monomials)
            insertMonomial(m);
    }

    void insertMonomial(const monomial_t& m) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), m, monomial_cmp);
        monomials.insert(it, m);
    }

    std::string str() const {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

    std::ostream& print(std::ostream& os) const override;

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly) {
        return poly.print(os);
    }

    int compare(const CasNode* other) const override {
        assert(false && "Do not compare Polynomial");
        return -2;
    }

    bool equals(const CasNode* other) const override {
        assert(false && "Unimplemented yet");
        return false;
    }

    expr_value getExprValue() const override {
        std::complex<double> v = 0.0;
        std::complex<double> mV = 1.0;
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

    Polynomial toPolynomial() override { return Polynomial(*this); };

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


class Context {
    std::vector<VariableNode*> vars;
    std::vector<ConstantNode*> consts;
    std::vector<CasNode*> nodes;
public:
    Context(int nparams = 0)
        : vars(nparams),
          consts({ new ConstantNode(0.0), new ConstantNode(1.0), new ConstantNode(-1.0) }),
          nodes() {
        for (unsigned i = 0; i < nparams; i++)
            vars[i] = new VariableNode(i);
    }

    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    ~Context() {
        for (auto& n : vars)
            delete(n);
        for (auto& n : consts)
            delete(n);
        for (auto& n : nodes)
            delete(n);
    }

    VariableNode* getVar(int v) {
        for (auto it = vars.begin(); it != vars.end(); it++) {
            if (v == (*it)->v)
                return *it;
        }
        auto* var = new VariableNode(v);
        vars.push_back(var);
        return var;
    }

    ConstantNode* getConst(const std::complex<double>& value) {
        for (auto it = consts.begin(); it != consts.end(); it++) {
            if ((*it)->value == value)
                return *it;
        }
        auto* cons = new ConstantNode(value);
        consts.push_back(cons);
        return cons;
    }

    CosineNode* createCosNode(CasNode* node) {
        auto* n = new CosineNode(node);
        nodes.push_back(n);
        return n;
    }

    SineNode* createSinNode(CasNode* node) {
        auto* n = new SineNode(node);
        nodes.push_back(n);
        return n;
    }

    AddNode* createAddNode(CasNode* lhs, CasNode* rhs) {
        auto* n = new AddNode(lhs, rhs);
        nodes.push_back(n);
        return n;
    }

    ComplexExpNode* createCompExpNode(CasNode* node) {
        auto* n = new ComplexExpNode(node);
        nodes.push_back(n);
        return n;
    }
};

} // namespace quench::cas

#endif // QUENCH_POLYNOMIAL_H