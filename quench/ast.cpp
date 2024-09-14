#include "quench/ast.h"
#include "quench/CircuitGraph.h"

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
        for (int i = 0; i < params.size()-1; i++)
            params[i].print(os) << " ";
        params.back().print(os) << ")";
    }

    os << " ";
    for (int i = 0; i < qubits.size()-1; i++)
        os << qubits[i] << " ";
    os << qubits.back();
    return os;
}

std::ostream& GateChainStmt::print(std::ostream& os) const {
    size_t size = gates.size();
    if (size == 0)
        return os;
    os << "  ";
    for (size_t i = 0; i < size-1; i++)
        gates[i].print(os) << "\n@ ";
    gates[size-1].print(os) << ";\n";
    return os;
}

std::ostream& CircuitStmt::print(std::ostream& os) const {
    os << "circuit<nqubits=" << nqubits << ", nparams=" << nparams << "> "
       << name << " {\n";
    for (const auto& s : stmts)
        s->print(os);
    return os << "}\n";
}

std::ostream& ParameterDefStmt::print(std::ostream& os) const {
    os << "#" << refNumber << " = { ";
    assert(gateMatrix.isParametrizedMatrix());

    for (const auto& poly : gateMatrix.matrix.parametrizedMatrix.data)
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

using namespace quench::circuit_graph;
CircuitGraph RootNode::toCircuitGraph() const {
    CircuitGraph graph;
    for (const auto& s : circuit.stmts) {
        
    }
    return graph;
}