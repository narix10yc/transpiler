#include "quench/ast.h"
#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

using namespace quench::ast;
using namespace quench::circuit_graph;

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
        for (unsigned i = 0; i < params.size()-1; i++)
            params[i].print(os) << " ";
        params.back().print(os) << ")";
    }

    os << " ";
    for (unsigned i = 0; i < qubits.size()-1; i++)
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
    for (const auto& gate : chain.gates) {
        // update number of qubits
        for (const auto& q : gate.qubits) {
            if (q >= nqubits)
                nqubits = q + 1;
        }
    }
}

QuantumGate RootNode::gateApplyToQuantumGate(const GateApplyStmt& gateApplyStmt) {
    if (gateApplyStmt.paramRefNumber >= 0) {
        for (auto it = paramDefs.begin(); it != paramDefs.end(); it++) {
            if (it->refNumber == gateApplyStmt.paramRefNumber)
                return QuantumGate(it->gateMatrix, gateApplyStmt.qubits);
        }
        assert(false && "Cannot find parameter def stmt");
        return QuantumGate();
    }
    return QuantumGate(
            GateMatrix::FromParameters(
                gateApplyStmt.name, gateApplyStmt.params, casContext),
            gateApplyStmt.qubits);
}

CircuitGraph RootNode::toCircuitGraph() {
    CircuitGraph graph;
    for (const auto& s : circuit.stmts) {
        const GateChainStmt* chain = dynamic_cast<const GateChainStmt*>(s.get());
        if (chain == nullptr) {
            std::cerr << Color::YELLOW_FG << Color::BOLD << "Warning: " << Color::RESET
                      << "Unable to convert to GateChainStmt when calling RootNode::toCircuitGraph\n";
            continue;
        }
        if (chain->gates.empty())
            continue;
        
        auto quGate = gateApplyToQuantumGate(chain->gates[0]);
        for (int i = 1; i < chain->gates.size(); i++)
            quGate = quGate.lmatmul(gateApplyToQuantumGate(chain->gates[i]));
        graph.addGate(quGate);
    }
    return graph;
}
