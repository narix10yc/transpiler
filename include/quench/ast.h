#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>

namespace quench::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual std::ostream& print(std::ostream& os) const = 0;
};

class Expression : public Node {

};

class PolynomialExpr;

class CASExpr : Expression {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual expr_value getExprValue() const = 0;

    /// @brief Compare with another node. Notice that compare == 0 does 'not' 
    /// mean equal. This includes:
    /// exponent of the PowerExpr class
    /// coefficient of the MonomialExpr class
    /// @return -1: less than; 0: equal; +1: greater than
    virtual int compare(const CASExpr*) const = 0;

    virtual bool equals(const CASExpr*) const = 0;

    bool equals(std::shared_ptr<CASExpr> p) const {
        return equals(p.get());
    }

    virtual inline int getSortPriority() const = 0;

    virtual PolynomialExpr toPolynomial() const = 0;

    virtual ~CASExpr() = default;
};

class BasicCASExpr : public CASExpr {
public:
    std::ostream& print(std::ostream& os) const override {
        assert(false && "BasicCASExpr");
        return os;
    }
};

class ConstantExpr : public BasicCASExpr {
    double value;
public:
    ConstantExpr(double value) : value(value) {}
    
    std::ostream& print(std::ostream& os) const override {
        return os << value;
    }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherConstantExpr = dynamic_cast<const ConstantExpr*>(other);
        assert(otherConstantExpr != nullptr);
        if (value < otherConstantExpr->value)
            return -1;
        if (value == otherConstantExpr->value)
            return 0;
        return +1;
    }

    bool equals(const CASExpr* other) const override {
        if (auto otherConstantExpr = dynamic_cast<const ConstantExpr*>(other))
            return (otherConstantExpr->value == value);
        return false;
    }

    inline int getSortPriority() const override { return 0; }

    PolynomialExpr toPolynomial() const override;
};

class VariableExpr : public BasicCASExpr {
    std::string name;
public:
    VariableExpr(const std::string& name) : name(name) {}

    std::ostream& print(std::ostream& os) const override {
        return os << name;
    }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherVariableExpr = dynamic_cast<const VariableExpr*>(other);
        assert(otherVariableExpr != nullptr);

        return name.compare(otherVariableExpr->name);
    }

    bool equals(const CASExpr* other) const override {
        if (auto otherVariableExpr = dynamic_cast<const VariableExpr*>(other))
            return (otherVariableExpr->name == name);
        return false;
    }

    inline int getSortPriority() const override { return 10; }

    PolynomialExpr toPolynomial() const override;
};

class CosineExpr : public BasicCASExpr {
    std::shared_ptr<BasicCASExpr> node;
public:
    CosineExpr(std::shared_ptr<BasicCASExpr> node) : node(node) {}

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

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosineExpr = dynamic_cast<const CosineExpr*>(other);
        assert(otherCosineExpr != nullptr);
        return node->compare(otherCosineExpr->node.get());
    }

    bool equals(const CASExpr* other) const override {
        auto otherCosineExpr = dynamic_cast<const CosineExpr*>(other);
        if (otherCosineExpr == nullptr)
            return false;
        return (node->equals(otherCosineExpr->node.get()));
    }

    inline int getSortPriority() const override { return 20; }

    PolynomialExpr toPolynomial() const override;

};

class SineExpr : public BasicCASExpr {
    std::shared_ptr<BasicCASExpr> node;
public:
    SineExpr(std::shared_ptr<BasicCASExpr> node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "sin(";
        node->print(os);
        os << ")";
        return os;
    }

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSineExpr = dynamic_cast<const SineExpr*>(other);
        assert(otherSineExpr != nullptr);
        return node->compare(otherSineExpr->node.get());
    }

    bool equals(const CASExpr* other) const override {
        auto otherSineExpr = dynamic_cast<const SineExpr*>(other);
        if (otherSineExpr == nullptr)
            return false;
        return (node->equals(otherSineExpr->node.get()));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    inline int getSortPriority() const override { return 30; }

    PolynomialExpr toPolynomial() const override;

};

class PowerExpr : public CASExpr {
public:
    std::shared_ptr<BasicCASExpr> base;
    int exponent;
    PowerExpr(std::shared_ptr<BasicCASExpr> base, int exponent = 1)
        : base(base), exponent(exponent) {}

    std::ostream& print(std::ostream& os) const override {
        base->print(os);
        if (exponent != 1.0)
            os << "**" << exponent;
        return os;
    }

    int compare(const PowerExpr& other) const {
        return base->compare(other.base.get());
    }

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherPowerExpr = dynamic_cast<const PowerExpr*>(other);
        assert(otherPowerExpr != nullptr);
        return compare(*otherPowerExpr);
    }

    bool operator==(const PowerExpr& other) const {
        if (exponent != other.exponent)
            return false;
        return base->equals(other.base.get());
    }

    bool operator!=(const PowerExpr& other) const {
        if (exponent != other.exponent)
            return true;
        return !(base->equals(other.base.get()));
    }

    bool equals(const CASExpr* other) const override {
        auto otherPowerExpr = dynamic_cast<const PowerExpr*>(other);
        if (otherPowerExpr == nullptr)
            return false;
        return (*this) == (*otherPowerExpr);
    }

    expr_value getExprValue() const override {
        auto baseValue = base->getExprValue();
        if (!baseValue.isConstant)
            return { false };
        return { true, std::pow(baseValue.value, exponent) };
    }

    inline int getSortPriority() const override { return 40; }

    PolynomialExpr toPolynomial() const override;

};

class MonomialExpr : public CASExpr {
public:
    double coefficient;
    std::vector<PowerExpr> pows;

    MonomialExpr(double coef) : coefficient(coef), pows() {}
    MonomialExpr(std::shared_ptr<BasicCASExpr> basicCASExpr, double coef = 1.0) 
        : coefficient(coef), pows({PowerExpr{basicCASExpr}}) {}
    MonomialExpr(std::initializer_list<PowerExpr> pows, double coef = 1.0)
        : coefficient(coef), pows(pows) {}

    int degree() const {
        int s = 0;
        for (const auto& pow : pows)
            s += pow.exponent;
        return s;
    }

    void sortSelf() {
        std::sort(pows.begin(), pows.end(),
            [](const PowerExpr& a, const PowerExpr& b) { return a.compare(b) < 0; });
    }

    MonomialExpr sortAndSimplify() const {
        MonomialExpr newMonomialExpr(coefficient);
        // this makes sure no replicates
        for (const auto& pow : pows)
            newMonomialExpr *= pow;
        newMonomialExpr.sortSelf();
        return newMonomialExpr;
    }

    std::ostream& print(std::ostream& os) const override {
        auto length = pows.size();
        if (length == 0) {
            os << coefficient;
            return os;
        }

        if (coefficient == -1.0)
            os << "-";
        else if (coefficient != 1.0)
            os << coefficient;
        
        for (unsigned i = 0; i < length-1; i++) {
            pows[i].print(os);
            os << " ";
        }
        pows[length-1].print(os);
        return os;
    }

    int compare(const MonomialExpr& other) const {
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

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherMonomialExpr = dynamic_cast<const MonomialExpr*>(other);
        assert(otherMonomialExpr != nullptr);
        return compare(*otherMonomialExpr);
    }

    bool equals(const CASExpr* other) const override {
        auto otherMonomialExpr = dynamic_cast<const MonomialExpr*>(other);
        if (otherMonomialExpr == nullptr)
            return false;
        if (otherMonomialExpr->coefficient != coefficient)
            return false;
        if (otherMonomialExpr->pows.size() != pows.size())
            return false;

        auto thisIt = pows.begin();
        auto otherIt = otherMonomialExpr->pows.begin();
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

    bool operator==(const MonomialExpr& other) const {
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

    bool operator!=(const MonomialExpr& other) const {
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

    MonomialExpr& operator*=(const PowerExpr& p) {
        for (auto& pow : pows) {
            if (pow.compare(p) == 0) {
                pow.exponent += p.exponent;
                return *this;
            }
        }
        pows.push_back(p);
        return *this;
    }

    MonomialExpr operator*(const PowerExpr& other) const {
        MonomialExpr newMonomialExpr(*this);
        return newMonomialExpr *= other;
    }

    MonomialExpr& operator*=(const MonomialExpr& other) {
        for (const auto& pow : other.pows)
            (*this) *= pow;
        this->coefficient *= other.coefficient;
        return *this;
    }

    MonomialExpr operator*(const MonomialExpr& other) const {
        MonomialExpr newMonomialExpr(*this);
        return newMonomialExpr *= other;
    }

    PolynomialExpr toPolynomial() const override;
};

class PolynomialExpr : public CASExpr {
public:
    std::vector<MonomialExpr> monomials;
    
    PolynomialExpr() : monomials() {}
    PolynomialExpr(std::initializer_list<MonomialExpr> monomials)
        : monomials(monomials) {}

    void sortSelf() {
        std::sort(monomials.begin(), monomials.end(),
            [](const MonomialExpr& a, const MonomialExpr& b) {
                return a.compare(b) < 0;
            });
    }

    PolynomialExpr sortAndSimplify() const {
        PolynomialExpr newPoly;
        // this makes sure no replicates
        for (const auto& monomial : monomials)
            newPoly += monomial.sortAndSimplify();
        newPoly.sortSelf();
        return newPoly;
    }

    std::ostream& print(std::ostream& os) const override {
        auto size = monomials.size();
        if (size == 0)
            return os;

        for (unsigned i = 0; i < size-1; i++) {
            monomials[i].print(os);
            os << " + ";
        }
        monomials[size-1].print(os);
        return os;
    }

    int compare(const PolynomialExpr& other) const {
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

    int compare(const CASExpr* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;

        auto otherPolynomialExpr = dynamic_cast<const PolynomialExpr*>(other);
        assert(otherPolynomialExpr != nullptr);
        return compare(*otherPolynomialExpr);
    }

    bool operator==(const PolynomialExpr& other) const {
        if (monomials.size() != other.monomials.size())
            return false;
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i] != other.monomials[i])
                return false;
        }
        return true;
    }

    bool operator!=(const PolynomialExpr& other) const {
        if (monomials.size() != other.monomials.size())
            return true;
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i] != other.monomials[i])
                return true;
        }
        return false;
    }

    bool equals(const CASExpr* other) const override {
        auto otherPolynomialExpr = dynamic_cast<const PolynomialExpr*>(other);
        if (otherPolynomialExpr == nullptr)
            return false;
        return (*this) == (*otherPolynomialExpr);
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

    PolynomialExpr& operator+=(const MonomialExpr& m) {
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (monomials[i].compare(m) == 0) {
                monomials[i].coefficient += m.coefficient;
                return *this;
            }
        }
        monomials.push_back(m);
        return *this;
    }

    PolynomialExpr& operator+=(const PolynomialExpr& other) {
        for (const auto& monomial : other.monomials)
            (*this) += monomial;
        return *this;
    }

    PolynomialExpr operator+(const PolynomialExpr& other) const {
        PolynomialExpr newPoly(*this);
        return newPoly += other;
    }

    PolynomialExpr& operator*=(const MonomialExpr& m) {
        for (auto& monomial : monomials)
            monomial *= m;
        return *this;
    }

    PolynomialExpr operator*(const MonomialExpr& m) const {
        PolynomialExpr newPoly(*this);
        return newPoly *= m;
    }

    PolynomialExpr& operator*=(const PolynomialExpr& other) {
        return (*this) = (*this) * other;
    }

    PolynomialExpr operator*(const PolynomialExpr& other) const {
        PolynomialExpr newPoly;
        for (const auto& m : other.monomials)
            newPoly += (*this) * m;
        return newPoly;
    }

    inline int getSortPriority() const override { return 60; }

    PolynomialExpr toPolynomial() const override { return *this; }
};


class Statement : public Node {

};

class RootNode : public Node {
public:
    std::vector<std::unique_ptr<Statement>> stmts;

    RootNode() : stmts() {}

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int>
class ParameterRefExpr : public Expression {
public:
    int number;

    ParameterRefExpr(int number) : number(number) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "#" << number;
    }
};

class GateApplyStmt : public Statement {
public:
    std::string name;
    std::vector<int> qubits;
    std::vector<std::shared_ptr<CASExpr>> parameters;
    int paramReference;

    GateApplyStmt(const std::string& name)
        : name(name), qubits(), parameters(), paramReference(-1) {} 

    std::ostream& print(std::ostream& os) const override;
};

class ParameterDefStmt : public Statement {

};

class CircuitStmt : public Statement {
public:
    std::string name;
    int nqubits;
    std::vector<std::unique_ptr<Statement>> stmts;
    std::vector<VariableExpr> parameters;
    
    CircuitStmt() : nqubits(0), stmts() {}

    void addGate(std::unique_ptr<GateApplyStmt> gate);

    std::ostream& print(std::ostream& os) const override;
};



} // namespace quench::ast

#endif // QUENCH_AST_H