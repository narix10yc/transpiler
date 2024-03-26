#ifndef AST_H_
#define AST_H_

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "utils.h"

namespace ast {

class Node;
class RootNode;

class Statement;

class Expression;
class NumericExpr;
class VariableExpr;
class UnaryExpr;
class BinaryExpr;


class Node {
public:
    virtual ~Node() = default;
    virtual std::string ToString() const = 0;
    virtual void prettyPrint(std::ofstream& f, int depth) const = 0;
};


class Statement : public Node {
public:
    std::string ToString() const { return "Stmt"; }
    virtual void prettyPrint(std::ofstream& f, int depth) const {}
};


class RootNode : public Node {
    std::vector<std::unique_ptr<Statement>> stmts;
public:
    std::string ToString() const { return "Root"; }
    void addStmt(std::unique_ptr<Statement> stmt) {
        stmts.push_back(std::move(stmt));
    }
    virtual void prettyPrint(std::ofstream& f, int depth) const;
};


class Expression : public Node {
public:
    virtual std::string ToString() const { return "Expr"; }
    virtual void prettyPrint(std::ofstream& f, int depth) const {}
};


class NumericExpr : public Expression {
    double value;
public:
    NumericExpr(double value) : value(value) {}
    virtual std::string ToString() const override 
    { return "Numeric(" + std::to_string(value) + ")"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;

    double getValue() const { return value; }
};


class VariableExpr : public Expression {
    std::string name;
public:
    VariableExpr(std::string name) : name(name) {}
    virtual std::string ToString() const override
    { return "Variable(" + name + ")"; }

    virtual void prettyPrint(std::ofstream& f, int depth) const override;

    std::string getName() const { return name; }
};


class UnaryExpr : public Expression {
    UnaryOp op;
    std::unique_ptr<Expression> expr;
public:
    UnaryExpr(UnaryOp op, std::unique_ptr<Expression> expr)
        : op(op), expr(std::move(expr)) {}
    virtual std::string ToString() const override { return ""; }
    virtual void prettyPrint(std::ofstream& f, int depth) const override;
};


class BinaryExpr : public Expression {
    BinaryOp op;
    std::unique_ptr<Expression> lhs, rhs;
public:
    BinaryExpr(BinaryOp op, 
               std::unique_ptr<Expression> lhs, 
               std::unique_ptr<Expression> rhs)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    virtual std::string ToString() const override { return ""; }
    virtual void prettyPrint(std::ofstream& f, int depth) const override;

    const Expression& getLHS() const { return *lhs; }
    const Expression& getRHS() const { return *rhs; }
    const BinaryOp getOp() const { return op; }
};


class IfThenElseStmt : public Statement {
    std::unique_ptr<Expression> ifExpr; 
    std::vector<std::unique_ptr<Statement>> thenBody;
    std::vector<std::unique_ptr<Statement>> elseBody;
public:
    IfThenElseStmt(std::unique_ptr<Expression> ifExpr) 
        : ifExpr(std::move(ifExpr)) {}

    virtual std::string ToString() const override 
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
    
    virtual void prettyPrint(std::ofstream& f, int depth) const override;
};


} // end of namespace ast

#endif // AST_H_
