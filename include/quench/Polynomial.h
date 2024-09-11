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

class CASNode {
public:
    struct expr_value {
        bool isConstant;
        std::complex<double> value;
    };

    virtual std::ostream& print(std::ostream&) const = 0;

    virtual std::ostream& printLaTeX(std::ostream&) const = 0;

    virtual expr_value getExprValue() const = 0;

    virtual int compare(const CASNode* other) const = 0;

    virtual bool equals(const CASNode*) const = 0;

    virtual int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() = 0;

    virtual ~CASNode() = default;
};

class AtomicCASNode : public CASNode {};

class ConstantNode : public AtomicCASNode {
    std::complex<double> value;
public:
    ConstantNode(std::complex<double> value) : value(value) {}

    std::complex<double> getValue() const { return value; }
    
    std::ostream& print(std::ostream& os) const override {
        return os << value;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << value;
    }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const CASNode* other) const override {
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

    bool equals(const CASNode* other) const override {
        if (auto otherConstantNode = dynamic_cast<const ConstantNode*>(other))
            return (otherConstantNode->value == value);
        return false;
    }

    int getSortPriority() const override { return 0; }

    Polynomial toPolynomial() override;
};

class VariableNode : public AtomicCASNode {
    std::string name;
public:
    VariableNode(const std::string& name) : name(name) {}

    std::string getName() const { return name; }

    std::ostream& print(std::ostream& os) const override {
        return os << name;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << name;
    }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const CASNode* other) const override {
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

    Polynomial toPolynomial() override;
};

class CosineNode : public CASNode {
    CASNode* node;
public:
    CosineNode(CASNode* node) : node(node) {}

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

    int compare(const CASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        assert(otherCosineNode != nullptr);
        return node->compare(otherCosineNode->node);
    }

    bool equals(const CASNode* other) const override {
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        if (otherCosineNode == nullptr)
            return false;
        return (node->equals(otherCosineNode->node));
    }

    int getSortPriority() const override { return 20; }

    Polynomial toPolynomial() override;

};

class SineNode : public CASNode {
    CASNode* node;
public:
    SineNode(CASNode* node) : node(node) {}

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

    int compare(const CASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        assert(otherSineNode != nullptr);
        return node->compare(otherSineNode->node);
    }

    bool equals(const CASNode* other) const override {
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

class VarAddNode : public CASNode {
    VariableNode* lhs;
    VariableNode* rhs;
public:
    VarAddNode(VariableNode* lhs, VariableNode* rhs) : lhs(lhs), rhs(rhs) {}

    std::ostream& print(std::ostream& os) const override {
        os << "(";
        lhs->print(os);
        os << " + ";
        rhs->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "(";
        lhs->print(os);
        os << " + ";
        rhs->print(os);
        os << ")";
        return os;
    }

    int compare(const CASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherVarAddNode = dynamic_cast<const VarAddNode*>(other);
        assert(otherVarAddNode != nullptr);

        auto compareLHS = lhs->compare(otherVarAddNode->lhs);
        if (compareLHS != 0)
            return compareLHS;
        return rhs->compare(otherVarAddNode->rhs);
    }

    bool equals(const CASNode* other) const override {
        auto otherVarAddNode = dynamic_cast<const VarAddNode*>(other);
        if (otherVarAddNode == nullptr)
            return false;
        return (lhs->equals(otherVarAddNode->lhs))
                && (rhs->equals(otherVarAddNode->rhs));
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

class ComplexExpNode : public CASNode {
    CASNode* node;
public:
    ComplexExpNode(CASNode* node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "cexp(";
        node->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "\\exp(i";
        node->print(os);
        os << ")";
        return os;
    }

    int compare(const CASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCExpNode = dynamic_cast<const ComplexExpNode*>(other);
        assert(otherCExpNode != nullptr);
        return node->compare(otherCExpNode->node);
    }

    bool equals(const CASNode* other) const override {
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

    int getSortPriority() const override { return 35; }

    Polynomial toPolynomial() override;
};

class Polynomial : public CASNode {
public:
    struct monomial_t {
        struct power_t {
            CASNode* base;
            int exponent = 1;
        };
        std::complex<double> coef = { 1.0, 0.0 };
        std::vector<power_t> powers = {};

        int order() const {
            int sum = 0;
            for (const auto& p : powers)
                sum += p.exponent;
            return sum;
        }
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
    Polynomial(std::complex<double> v) : monomials({{v, {}}}) {}
    Polynomial(std::initializer_list<monomial_t> monomials)
        : monomials(monomials) {}

    void insertMonomial(const monomial_t& monomial) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
        monomials.insert(it, monomial);
    }

    std::string str() const {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

    std::ostream& print(std::ostream& os) const override;

    std::ostream& printLaTeX(std::ostream& os) const override;

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly) {
        return poly.print(os);
    }

    int compare(const CASNode* other) const override {
        assert(false && "Do not compare Polynomial");
        return -2;
    }

    bool equals(const CASNode* other) const override {
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
    std::vector<CASNode*> nodes;
public:
    Context(int nparams = 0)
        : vars(nparams),
          consts({ new ConstantNode(0.0), new ConstantNode(1.0), new ConstantNode(-1.0) }),
          nodes() {
        for (int i = 0; i < nparams; i++)
            vars[i] = new VariableNode("%" + std::to_string(i));
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

    VariableNode* getVar(const std::string& name) {
        for (auto it = vars.begin(); it != vars.end(); it++) {
            if ((*it)->getName() == name)
                return *it;
        }
        auto* var = new VariableNode(name);
        vars.push_back(var);
        return var;
    }

    ConstantNode* getConst(const std::complex<double>& value) {
        for (auto it = consts.begin(); it != consts.end(); it++) {
            if ((*it)->getValue() == value)
                return *it;
        }
        auto* cons = new ConstantNode(value);
        consts.push_back(cons);
        return cons;
    }

    CosineNode* createCosNode(CASNode* node) {
        auto* n = new CosineNode(node);
        nodes.push_back(n);
        return n;
    }

    SineNode* createSinNode(CASNode* node) {
        auto* n = new SineNode(node);
        nodes.push_back(n);
        return n;
    }

    VarAddNode* createAddNode(VariableNode* lhs, VariableNode* rhs) {
        auto* n = new VarAddNode(lhs, rhs);
        nodes.push_back(n);
        return n;
    }

    ComplexExpNode* createCompExpNode(CASNode* node) {
        auto* n = new ComplexExpNode(node);
        nodes.push_back(n);
        return n;
    }

};


} // namespace quench::cas

#endif // QUENCH_POLYNOMIAL_H