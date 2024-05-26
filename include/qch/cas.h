#ifndef QCH_CAS_H
#define QCH_CAS_H

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <cassert>
#include <utility>
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

    virtual bool equals(const Node*) const = 0;

    virtual inline int getSortPriority() const = 0;

    virtual Polynomial* canonicalize(CASContext&) const = 0;
};



class Constant : public Node {
    double value;
public:
    Constant(double value) : value(value) {}
    
    void print(std::ostream& os) const override { os << value; }

    expr_value getExprValue() const override {
        return { true, value };
    }

    bool equals(const Node* other) const override {
        if (auto otherConstant = dynamic_cast<const Constant*>(other))
            return (otherConstant->value == value);
        return false;
    }

    inline int getSortPriority() const override { return 0; }

    Polynomial* canonicalize(CASContext&) const override;
};


class Variable : public Node {
    std::string name;
public:
    Variable(const std::string& name) : name(name) {}

    void print(std::ostream& os) const override { os << name; }

    expr_value getExprValue() const override {
        return { false };
    }

    bool equals(const Node* other) const override {
        if (auto otherVariable = dynamic_cast<const Variable*>(other))
            return (otherVariable->name == name);
        return false;
    }

    inline int getSortPriority() const override { return 10; }

    Polynomial* canonicalize(CASContext&) const override;
};


class Cosine : public Node {
    Node* node;
public:
    Cosine(Node* node) : node(node) {}

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

    bool equals(const Node* other) const override {
        auto otherCosine = dynamic_cast<const Cosine*>(other);
        if (otherCosine == nullptr)
            return false;
        return (node->equals(otherCosine->node));
    }

    inline int getSortPriority() const override { return 20; }

    Polynomial* canonicalize(CASContext&) const override;
};


class Sine : public Node {
    Node* node;
public:
    Sine(Node* node) : node(node) {}

    void print(std::ostream& os) const override {
        os << "sin(";
        node->print(os);
        os << ")";
    }

    bool equals(const Node* other) const override {
        auto otherSine = dynamic_cast<const Sine*>(other);
        if (otherSine == nullptr)
            return false;
        return (node->equals(otherSine->node));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    inline int getSortPriority() const override { return 30; }

    Polynomial* canonicalize(CASContext&) const override;
};


class Power : public Node {
public:
    Node* base;
    int exponent;
    Power(Node* base, int exponent=1) : base(base), exponent(exponent) {}

    void print(std::ostream& os) const override {
        base->print(os);
        if (exponent != 1.0)
            os << "**" << exponent;
    }

    bool equals(const Node* other) const override {
        auto otherPower = dynamic_cast<const Power*>(other);
        if (otherPower == nullptr)
            return false;
        if (otherPower->exponent != exponent)
            return false;
        return (base->equals(otherPower->base));
    }

    expr_value getExprValue() const override {
        auto baseValue = base->getExprValue();
        if (!baseValue.isConstant)
            return { false };
        return { true, std::pow(baseValue.value, exponent) };
    }

    inline int getSortPriority() const override { return 40; }

    Polynomial* canonicalize(CASContext&) const override;
};

class Monomial : public Node {
public:
    double coefficient;
    std::vector<Power*> pows;

    Monomial(double coef) : coefficient(coef), pows() {}
    Monomial(double coef, Power* pow) : coefficient(coef), pows({pow}) {}
    Monomial(double coef, std::initializer_list<Power*> pows)
        : coefficient(coef), pows(pows) { sortAndSimplify(); }
    Monomial(double coef, const std::vector<Power*>& pows)
        : coefficient(coef), pows(pows) { sortAndSimplify(); }

    void sortAndSimplify() {
        if (pows.empty())
            return;

        std::sort(pows.begin(), pows.end(),
            [](const Power* a, const Power* b) -> bool {
                return a->base->getSortPriority() < b->base->getSortPriority();
            });
        
        std::vector<Power*> newPows;
        Power* p = pows[0];
        for (unsigned i = 1; i < pows.size(); i++) {
            if (p->base->equals(pows[i]->base))
                p->exponent += pows[i]->exponent;
            else {
                newPows.push_back(p);
                p = pows[i];
            }
        }
        newPows.push_back(p);
        pows = newPows;
    }

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
            os<< " * ";
        }
        pows[length-1]->print(os);
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
            if (!(pows[i]->equals(otherMonomial->pows[i])))
                return false;
        }
        return true;
    }

    expr_value getExprValue() const override {
        expr_value value { true, coefficient };
        expr_value powValue;
        for (auto* pow : pows) {
            powValue = pow->getExprValue();
            if (!powValue.isConstant)
                return { false };
            value.value *= powValue.value;
        }
        return value;
    }

    inline int getSortPriority() const override { return 50; }

    Polynomial* canonicalize(CASContext&) const override;
};


class Polynomial : public Node {
    struct monomial_cmp {
        bool operator() (const Monomial* a, const Monomial* b) const {
            return a->order() < b->order();
        }
    };
public:
    std::set<Monomial*, monomial_cmp> monomials;
    
    Polynomial() : monomials() {}
    Polynomial(Monomial* monomial) : monomials({monomial}) {}

    std::vector<Monomial*> getMonomialsVector() const {
        return { monomials.begin(), monomials.end() };
    }

    void print(std::ostream& os) const override {
        std::vector<Monomial*> vec(monomials.begin(), monomials.end());
        auto size = vec.size();
        if (size == 0)
            return;
        for (unsigned i = 0; i < size-1; i++) {
            vec[i]->print(os);
            os << " + ";
        }
        vec[size-1]->print(os);
    }

    bool equals(const Node* other) const override {
        auto otherPolynomial = dynamic_cast<const Polynomial*>(other);
        if (otherPolynomial == nullptr)
            return false;
        if (otherPolynomial->monomials.size() != monomials.size())
            return false;
        
        auto thisVec = getMonomialsVector();
        auto otherVec = otherPolynomial->getMonomialsVector();
        for (unsigned i = 0; i < monomials.size(); i++) {
            if (!(thisVec[i]->equals(otherVec[i])))
                return false;
        }
        return true;
    }

    expr_value getExprValue() const override {
        expr_value value { true, 0.0 };
        expr_value monomialValue;
        for (auto* monomial : monomials) {
            monomialValue = monomial->getExprValue();
            if (!monomialValue.isConstant)
                return { false };
            value.value += monomialValue.value;
        }
        return value;
    }

    inline int getSortPriority() const override { return 60; }

    Polynomial* canonicalize(CASContext&) const override;

    void scalarMulInPlace(double);

    void addInplace(Monomial*);
    void addInplace(Polynomial*);

    Polynomial operator+ (const Polynomial& other) const {
        Polynomial p(*this);
        for (auto monomial : other.monomials)
            p.addInplace(monomial);
        return p;
    }

};


class CASContext {
    std::vector<Node*> nodes;
public:
    CASContext() : nodes() {}

    size_t count() const { return nodes.size(); }

    template<typename T, typename... Args>
    T* get(Args&&... args) {
        T* node = new T(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    Constant* getConstant(double value) {
        return get<Constant>(value);
    }

    Variable* getVariable(const std::string& name) {
        return get<Variable>(name);
    }

    Cosine* getCosine(Node* node) {
        return get<Cosine>(node);
    }

    Sine* getSine(Node* node) {
        return get<Sine>(node);
    }

    Power* getPower(Node* base, int exponent=1) {
        return get<Power>(base, exponent);
    }

    Monomial* getMonomial(double coef, std::initializer_list<Power*> pows) {
        auto node = new Monomial(coef, pows);
        nodes.push_back(node);
        return node;
    }

};

} // namespace qch::cas


#endif // QCH_CAS_H