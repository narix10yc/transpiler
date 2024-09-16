#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <map>

#include "quench/QuantumGate.h"

namespace quench::circuit_graph {
    class CircuitGraph;
}

namespace quench::ast {


class CircuitCompatibleStmt {
public:
    virtual ~CircuitCompatibleStmt() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
};

class GateApplyStmt : public CircuitCompatibleStmt {
public:
    std::string name;
    std::vector<int> qubits;
    std::vector<quantum_gate::GateParameter> params;
    int paramRefNumber;

    GateApplyStmt(const std::string& name, int paramRefNumber = -1)
        : name(name), qubits(), paramRefNumber(paramRefNumber) {}

    GateApplyStmt(const std::string& name, int paramRefNumber,
                  std::initializer_list<int> qubits)
        : name(name), qubits(qubits), paramRefNumber(paramRefNumber) {} 

    std::ostream& print(std::ostream& os) const override;
};

class GateChainStmt : public CircuitCompatibleStmt {
public:
    std::vector<GateApplyStmt> gates;

    GateChainStmt() : gates() {}

    std::ostream& print(std::ostream& os) const override;
};

class QuantumCircuit {
public:
    std::string name;
    int nqubits;
    int nparams;
    std::vector<std::unique_ptr<CircuitCompatibleStmt>> stmts;
    
    QuantumCircuit() : nqubits(0), nparams(0), stmts() {}

    void addGateChain(const GateChainStmt& chain);

    std::ostream& print(std::ostream& os) const;
};

/// @brief '#'<number:int> '=' '{' ... '}'';'
class ParameterDefStmt {
public:
    int refNumber;
    quantum_gate::GateMatrix gateMatrix;

    ParameterDefStmt(int refNumber) : refNumber(refNumber), gateMatrix() {}

    std::ostream& print(std::ostream& os) const;
};

class RootNode {
public:
    QuantumCircuit circuit;
    std::vector<ParameterDefStmt> paramDefs;
    cas::Context casContext;

    RootNode() : circuit(), paramDefs(), casContext() {}

    std::ostream& print(std::ostream& os) const;

    quantum_gate::QuantumGate
    gateApplyToQuantumGate(const GateApplyStmt&);
    
    circuit_graph::CircuitGraph toCircuitGraph();
};


} // namespace quench::ast

#endif // QUENCH_AST_H