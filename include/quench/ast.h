#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>

#include "GateMatrix.h"

namespace quench::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual std::ostream& print(std::ostream& os) const = 0;
};

class Expression : public Node {};

/// @brief '#'<number:int>
class ParameterRefExpr : public Expression {
public:
    int number;

    ParameterRefExpr(int number) : number(number) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "#" << number;
    }
};


class Statement : public Node {};

class GateApplyStmt : public Statement {
public:
    std::string name;
    std::vector<int> qubits;
    cas::SquareComplexMatrix<cas::Polynomial> matrix;
    int paramReference;

    GateApplyStmt(const std::string& name)
        : name(name), qubits(), matrix(), paramReference(-1) {} 

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'
class ParameterDefStmt : public Statement {
public:
    std::unique_ptr<ParameterRefExpr> lhs;

};

class CircuitStmt : public Statement {
public:
    std::string name;
    int nqubits;
    std::vector<std::unique_ptr<Statement>> stmts;
    std::vector<std::shared_ptr<cas::VariableNode>> parameters;
    
    CircuitStmt() : nqubits(0), stmts() {}

    void addGate(std::unique_ptr<GateApplyStmt> gate);

    std::ostream& print(std::ostream& os) const override;
};


class RootNode : public Node {
public:
    std::vector<std::unique_ptr<Statement>> stmts;

    RootNode() : stmts() {}

    std::ostream& print(std::ostream& os) const override;
};


} // namespace quench::ast

#endif // QUENCH_AST_H