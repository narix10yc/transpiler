#ifndef SAOT_POLYNOMIAL_H
#define SAOT_POLYNOMIAL_H

#include <vector>
#include <string>
#include <iostream>

namespace saot::polynomial {

class CASContext;

class Node {
public:
    virtual std::ostream& print(std::ostream&) const = 0;

    virtual Node* derivative(
            const std::string& var, CASContext& ctx) const = 0;

    virtual Node* simplify(CASContext& ctx) { return this; }
};

class Atom : public Node {};

class Expression : public Node {};

class Numerics : public Atom {
public:
    double value;
    Numerics(double value) : value(value) {}

    std::ostream& print(std::ostream& os) const override {
        return os << value << " ";
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
};

class Variable : public Atom {
public:
    std::string name;
    Variable(const std::string& name) : name(name) {}

    std::ostream& print(std::ostream& os) const override {
        return os << name << " ";
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
};

class AddExpr : public Expression {
public:
    Node* lhs;
    Node* rhs;
    AddExpr(Node* lhs, Node* rhs)
        : lhs(lhs), rhs(rhs) {}
    
    std::ostream& print(std::ostream& os) const override {
        os << "+ ";
        lhs->print(os);
        return rhs->print(os);
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
    Node* simplify(CASContext& ctx) override;
};

class SubExpr : public Expression {
public:
    Node* lhs;
    Node* rhs;
    SubExpr(Node* lhs, Node* rhs)
        : lhs(lhs), rhs(rhs) {}
    
    std::ostream& print(std::ostream& os) const override {
        os << "- ";
        lhs->print(os);
        return rhs->print(os);
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
    Node* simplify(CASContext& ctx) override;
};

class MulExpr : public Expression {
public:
    Node* lhs;
    Node* rhs;
    MulExpr(Node* lhs, Node* rhs)
        : lhs(lhs), rhs(rhs) {}
    
    std::ostream& print(std::ostream& os) const override {
        os << "* ";
        lhs->print(os);
        return rhs->print(os);
        return os;
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
    Node* simplify(CASContext& ctx) override;
};

class CosExpr : public Expression {
public:
    Node* node;
    CosExpr(Node* node) : node(node) {}
    
    std::ostream& print(std::ostream& os) const override {
        os << "cos ";
        return node->print(os);
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
    Node* simplify(CASContext& ctx) override;
};

class SinExpr : public Expression {
public:
    Node* node;
    SinExpr(Node* node) : node(node) {}
    
    std::ostream& print(std::ostream& os) const override {
        os << "sin ";
        return node->print(os);
    }

    Node* derivative(const std::string& var, CASContext& ctx) const override;
    Node* simplify(CASContext& ctx) override;
};



class CASContext {
    std::vector<Variable*> vars;
    std::vector<Numerics*> nums;
    std::vector<Node*> nodes;
public:
    CASContext() : vars(), nums() {}
    
    Variable* getVariable(const std::string& name) {
        for (auto it = vars.begin(); it != vars.end(); it++) {
            if ((*it)->name == name)
                return *it;
        }
        auto var = new Variable(name);
        vars.push_back(var);
        return var;
    }

    Numerics* getNumerics(double value) {
        for (auto it = nums.begin(); it != nums.end(); it++) {
            if ((*it)->value == value)
                return *it;
        }
        auto num = new Numerics(value);
        nums.push_back(num);
        return num;
    }

    CosExpr* createCos(Node* node) {
        auto cos = new CosExpr(node);
        nodes.push_back(cos);
        return cos;
    }

    SinExpr* createSin(Node* node) {
        auto sin = new SinExpr(node);
        nodes.push_back(sin);
        return sin;
    }

    AddExpr* createAdd(Node* lhs, Node* rhs) {
        auto add = new AddExpr(lhs, rhs);
        nodes.push_back(add);
        return add;
    }

    SubExpr* createSub(Node* lhs, Node* rhs) {
        auto sub = new SubExpr(lhs, rhs);
        nodes.push_back(sub);
        return sub;
    }

    MulExpr* createMul(Node* lhs, Node* rhs) {
        auto mul = new MulExpr(lhs, rhs);
        nodes.push_back(mul);
        return mul;
    }

};

} // namespace saot::polynomial

#endif // SAOT_POLYNOMIAL_H 