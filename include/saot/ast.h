#ifndef SAOT_AST_H
#define SAOT_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <map>

#include "saot/QuantumGate.h"

namespace saot::circuit_graph {
    class CircuitGraph;
}

namespace saot::ast {

class CircuitCompatibleStmt {
public:
    virtual ~CircuitCompatibleStmt() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
};

class MeasureStmt : public CircuitCompatibleStmt {
public:
    std::vector<int> qubits;
    MeasureStmt(std::initializer_list<int> qubits) : qubits(qubits) {}
    MeasureStmt(const std::vector<int>& qubits) : qubits(qubits) {}

    std::ostream& print(std::ostream& os) const override;
};

class GateApplyStmt : public CircuitCompatibleStmt {
public:
    std::string name;
    std::vector<int> qubits;
    int paramRefNumber;
    std::vector<quantum_gate::GateParameter> gateParams;

    GateApplyStmt(
            const std::string& name, const std::vector<int>& qubits,
            int paramRefNumber = -1)
        : name(name), qubits(qubits), paramRefNumber(paramRefNumber), gateParams() {}

    GateApplyStmt(
            const std::string& name, const std::vector<int>& qubits,
            const std::vector<quantum_gate::GateParameter>& gateParams)
        : name(name), qubits(qubits), paramRefNumber(-1), gateParams(gateParams) {}

    std::ostream& print(std::ostream& os) const override;
};

class GateChainStmt : public CircuitCompatibleStmt {
public:
    std::vector<GateApplyStmt> gates;

    GateChainStmt() : gates() {}

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'';'
class ParameterDefStmt {
public:
    int refNumber;
    quantum_gate::GateMatrix gateMatrix;

    ParameterDefStmt(int refNumber) : refNumber(refNumber), gateMatrix() {}

    std::ostream& print(std::ostream& os) const;
};

class QuantumCircuit {
public:
    std::string name;
    int nqubits;
    int nparams;
    std::vector<std::unique_ptr<CircuitCompatibleStmt>> stmts;
    std::vector<ParameterDefStmt> paramDefs;

    QuantumCircuit(const std::string& name = "qc") : name(name), nqubits(0), nparams(0), stmts(), paramDefs() {}

    std::ostream& print(std::ostream& os) const;

    quantum_gate::QuantumGate
    gateApplyToQuantumGate(const GateApplyStmt&);

    circuit_graph::CircuitGraph toCircuitGraph();
};



} // namespace saot::ast

#endif // SAOT_AST_H