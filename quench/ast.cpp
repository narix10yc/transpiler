#include "quench/ast.h"

using namespace quench::ast;

std::ostream& RootNode::print(std::ostream& os) const {
    circuit.print(os);
    os << "\n";

    os << "Warning: printing ParamDefStmt not implemented yet\n";

    return os;
}

std::ostream& GateApplyStmt::print(std::ostream& os) const {
    os << name;
    return os;
}

std::ostream& CircuitStmt::print(std::ostream& os) const {
    os << "circuit\n";
    return os;
}

void CircuitStmt::addGateChain(const GateChainStmt& chain) {
    stmts.push_back(std::make_unique<GateChainStmt>(chain));
    // update number of qubits
    for (const auto& gate : chain.gates) {
        for (const auto& q : gate.qubits) {
            if (q >= nqubits)
                nqubits = q + 1;
        }
    }
}
