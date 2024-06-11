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
class Statement : public Node {};

class GateApplyStmt : public Statement {
public:
    std::string name;
    std::vector<int> qubits;
    int paramRefNumber;

    GateApplyStmt(const std::string& name, int paramRefNumber = -1)
        : name(name), qubits(), paramRefNumber(paramRefNumber) {}

    GateApplyStmt(const std::string& name, int paramRefNumber,
                  std::initializer_list<int> qubits)
        : name(name), qubits(qubits), paramRefNumber(paramRefNumber) {} 

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'
class ParameterDefStmt : public Statement {
public:
    int refNumber;
    int nqubits;
    using GateMatrix = cas::SquareComplexMatrix<cas::Polynomial>;
    GateMatrix matrix;

    ParameterDefStmt(int refNumber)
        : refNumber(refNumber), nqubits(0), matrix() {}

    std::ostream& print(std::ostream& os) const override {return os;}

};

class GateChainStmt : public Statement {
public:
    std::vector<GateApplyStmt> gates;

    GateChainStmt() : gates() {}

    std::ostream& print(std::ostream& os) const override {
        auto it = gates.begin();
        while (it != gates.end()) {
            os << ((it == gates.begin()) ? "  " : "@ ");
            it->print(os) << "\n";
        }
        return os;
    }
};

class BlockOfGatesStmt : public Statement {
public:
    std::vector<GateChainStmt> chains;

    BlockOfGatesStmt() : chains() {}
    
    std::ostream& print(std::ostream& os) const override {
        for (const auto& chain : chains)
            chain.print(os);
        return os;
    }
};

class CircuitStmt : public Statement {
public:
    std::string name;
    int nqubits;
    std::vector<std::unique_ptr<Statement>> stmts;
    std::vector<std::shared_ptr<cas::VariableNode>> parameters;
    
    CircuitStmt() : nqubits(0), stmts() {}

    void addGateChain(const GateChainStmt& chain);

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