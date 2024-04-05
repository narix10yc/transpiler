#ifndef AST_H_
#define AST_H_


#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "token.h"

namespace simulation {
class CPUGenContext;
class FPGAGenContext;
} // namespace simulation

namespace openqasm::ast {

class Node;
class RootNode;

class Statement;

class Expression;
class NumericExpr;
class VariableExpr;
class UnaryExpr;
class BinaryExpr;

class ExpressionValue {
public:
    const bool isConstant;
    const double value;
    ExpressionValue(double value) : isConstant(true), value(value) {}
    ExpressionValue(bool isConstant) : isConstant(isConstant), value(0) {}
};


class Node {
public:
    virtual ~Node() = default;
    virtual std::string toString() const = 0;
    virtual void prettyPrint(std::ofstream& f, int depth) const = 0;
    virtual void genCPU(const simulation::CPUGenContext&) const {}
};


class Statement : public Node {
public:
    std::string toString() const override { return "Stmt"; }
    void prettyPrint(std::ofstream& f, int depth) const override {}
};


class RootNode : public Node {
    std::vector<std::unique_ptr<Statement>> stmts;
public:
    std::string toString() const override { return "Root"; }
    void prettyPrint(std::ofstream& f, int depth) const override;
    void addStmt(std::unique_ptr<Statement> stmt) {
        stmts.push_back(std::move(stmt));
    }
    
    size_t countStmts() { return stmts.size(); }
    Statement getStmt(size_t index) { return *(stmts[index]); }

    void genCPU(const simulation::CPUGenContext& ctx) const override {
        for (auto& stmt : stmts) {
            stmt->genCPU(ctx);
        }
    }
    
};


class Expression : public Node {
public:
    std::string toString() const override { return "Expr"; }
    void prettyPrint(std::ofstream& f, int depth) const override {}
    virtual ExpressionValue getExprValue() const { return false; }
};


class NumericExpr : public Expression {
    double value;
public:
    NumericExpr(double value) : value(value) {}
    std::string toString() const override {
        return "(" + std::to_string(value) + ")";
    }

    void prettyPrint(std::ofstream& f, int depth) const override;

    double getValue() const { return value; }
    virtual ExpressionValue getExprValue() const override { return value; }
};


class VariableExpr : public Expression {
    std::string name;
public:
    VariableExpr(std::string name) : name(name) {}
    
    std::string getName() const { return name; }
    
    std::string toString() const override {
        return "(" + name + ")";
    }
    void prettyPrint(std::ofstream& f, int depth) const override;

    virtual ExpressionValue getExprValue() const override {
        return (name == "pi") ? 3.14159265358979323846 : false;
    }

};

class SubscriptExpr : public Expression {
    std::string name;
    int index;
public:
    SubscriptExpr(std::string name, int index) : name(name), index(index) {}

    std::string getName() const { return name; }
    int getIndex() const { return index; }

    std::string toString() const override {
        return name + "[" + std::to_string(index) + "]";
    }
    void prettyPrint(std::ofstream& f, int depth) const override;

    virtual ExpressionValue getExprValue() const override { return false; }
};

class UnaryExpr : public Expression {
    UnaryOp op;
    std::unique_ptr<Expression> expr;
public:
    UnaryExpr(UnaryOp op, std::unique_ptr<Expression> expr)
        : op(op), expr(std::move(expr)) {}
    std::string toString() const override { return "UnaryExpr"; }
    void prettyPrint(std::ofstream& f, int depth) const override;
};


class BinaryExpr : public Expression {
    BinaryOp op;
    std::unique_ptr<Expression> lhs, rhs;
public:
    BinaryExpr(BinaryOp op, 
               std::unique_ptr<Expression> lhs, 
               std::unique_ptr<Expression> rhs)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    std::string toString() const override { return "BinaryExpr"; }
    void prettyPrint(std::ofstream& f, int depth) const override;

    const Expression getLHS() const { return *lhs; }
    const Expression getRHS() const { return *rhs; }
    BinaryOp getOp() const { return op; }
    virtual ExpressionValue getExprValue() const override {
        auto lhsValue = lhs->getExprValue();
        auto rhsValue = rhs->getExprValue();
        if (!lhsValue.isConstant || !rhsValue.isConstant) {
            return false;
        }
        // both are constant
        switch (op) {
            case BinaryOp::Add: return lhsValue.value + rhsValue.value;
            case BinaryOp::Sub: return lhsValue.value - rhsValue.value;
            case BinaryOp::Mul: return lhsValue.value * rhsValue.value;
            case BinaryOp::Div: return lhsValue.value / rhsValue.value;
            default: return false;
        }
    }
};


class IfThenElseStmt : public Statement {
    std::unique_ptr<Expression> ifExpr; 
    std::vector<std::unique_ptr<Statement>> thenBody;
    std::vector<std::unique_ptr<Statement>> elseBody;
public:
    IfThenElseStmt(std::unique_ptr<Expression> ifExpr) 
        : ifExpr(std::move(ifExpr)) {}

    virtual std::string toString() const override 
    { return "IfThenElseStmt"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;

    void addThenBody(std::unique_ptr<Statement> stmt)
    { thenBody.push_back(std::move(stmt)); }

    void addElseBody(std::unique_ptr<Statement> stmt)
    { elseBody.push_back(std::move(stmt)); }
};


class VersionStmt : public Statement {
    std::string version;
public:
    VersionStmt(std::string version) : version(version) {}
    std::string getVersion() const { return version; }

    virtual std::string toString() const override 
    { return "Version(" + version + ")"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;
};


class QRegStmt : public Statement {
    std::string name;
    int size;
public:
    QRegStmt(std::string name, int size) : name(name), size(size) {}

    std::string getName() const { return name; }
    int getSize() const { return size; }

    virtual std::string toString() const override 
    { return "QReg(" + name + ", " + std::to_string(size) + ")"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;
};

class CRegStmt : public Statement {
    std::string name;
    int size;
public:
    CRegStmt(std::string name, int size) : name(name), size(size) {}

    std::string getName() const { return name; }
    int getSize() const { return size; }

    virtual std::string toString() const override 
    { return "CReg(" + name + ", " + std::to_string(size) + ")"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;
};


class GateApplyStmt : public Statement {
    std::string name;
    std::vector<std::unique_ptr<Expression>> parameters;
    std::vector<std::unique_ptr<Expression>> targets;
public:
    GateApplyStmt(std::string name) : name(name) {}

    void addParameter(std::unique_ptr<Expression> param)
    { parameters.push_back(std::move(param)); }

    void addTarget(std::unique_ptr<Expression> targ)
    { targets.push_back(std::move(targ)); }

    std::string getName() const { return name; }

    size_t countParameters() const { return parameters.size(); }
    size_t countTargets() const { return targets.size(); }

    const Expression getParameter(size_t index) {
        return *(parameters[index]);
    }

    const Expression getTarget(size_t index) {
        return *(targets[index]);
    }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;

    // void genCPU(const simulation::CPUGenContext& ctx) const override;
};

} // namespace openqasm::ast

#endif // AST_H_
