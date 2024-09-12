#include "quench/ast.h"

using namespace quench::ast;

std::ostream& RootNode::print(std::ostream& os) const {
    circuit.print(os);
    os << "\n";

    for (const auto& def : paramDefs)
        def.print(os);

    return os;
}

std::ostream& GateApplyStmt::print(std::ostream& os) const {
    os << name;
    if (paramRefNumber >= 0)
        os << "(#" << paramRefNumber << ")";
    else if (!params.empty()) {
        os << "(";
        for (const auto& p : params)
            p.print(os) << " ";
        os << ")";
    }

    os << " ";
    for (const auto& q : qubits)
        os << q << " ";
    return os;
}

std::ostream& GateChainStmt::print(std::ostream& os) const {
    size_t size = gates.size();
    if (size == 0)
        return os;
    os << "  ";
    for (size_t i = 0; i < size-1; i++)
        gates[i].print(os) << "\n@ ";
    gates[size-1].print(os) << "\n";
    return os;
}

std::ostream& CircuitStmt::print(std::ostream& os) const {
    os << "circuit<" << nqubits << " qubits, " << nparams << " params> "
       << name << " {\n";
    for (const auto& s : stmts)
        s->print(os);
    return os << "}\n";
}

std::ostream& ParameterDefStmt::print(std::ostream& os) const {
    os << "#" << refNumber << " = { ";
    assert(matrix.isParametrizedMatrix());

    for (const auto& poly : matrix.pMatrix().data)
        poly.print(os) << ", ";

    return os << "}\n";
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
