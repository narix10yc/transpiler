#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>

namespace quench::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual std::ostream& print(std::ostream& os) const = 0;
};

class Expression : public Node {

};

class Statement : public Node {

};

class RootNode : public Node {
public:
    std::vector<std::unique_ptr<Statement>> stmts;

    RootNode() : stmts() {}

    std::ostream& print(std::ostream& os) const override {}
};

class GateApplyStmt : public Statement {
public:
    std::string name;
    std::vector<int> qubits;
    std::vector<double> parameters;
    int paramReference;
    GateApplyStmt(const std::string& name)
        : name(name), qubits(), parameters(), paramReference(-1) {} 
};

class CircuitStmt : public Statement {
public:
    int nqubits;
    std::vector<std::unique_ptr<Statement>> stmts;
    
    CircuitStmt() : nqubits(0), stmts() {}

    void addGate(std::unique_ptr<GateApplyStmt> gate);

    std::ostream& print(std::ostream& os) const override;
};



} // namespace quench::ast

#endif // QUENCH_AST_H