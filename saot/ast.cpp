#include "saot/ast.h"
#include "saot/CircuitGraph.h"
#include "utils/iocolor.h"

using namespace saot::ast;
using namespace saot::circuit_graph;
using namespace saot::quantum_gate;

template<typename T>
std::ostream& printVector(
        std::ostream& os, const std::vector<T>& vec,
        const std::string& sep = ",") {
    if (vec.empty())
        return os;
    auto it = vec.cbegin();
    os << (*it);
    while (++it != vec.cend())
        os << sep << (*it);
    return os;
}

std::ostream& MeasureStmt::print(std::ostream& os) const {
    os << "m ";
    printVector(os, qubits, " ");
    return os << "\n";
}

std::ostream& QuantumCircuit::print(std::ostream& os) const {
    os << "circuit<nqubits=" << nqubits << ", nparams=" << nparams << "> "
       << name << " {\n";
    for (const auto& s : stmts)
        s->print(os);
    os << "\n";
    for (const auto& def : paramDefs)
        def.print(os);

    return os;
}


std::ostream& GateApplyStmt::print(std::ostream& os) const {
    os << name;
    // parameter
    if (paramRefNumber >= 0)
        os << "(#" << paramRefNumber << ")";
    else if (!gateParams.empty()) {
        os << "(";
        auto it = gateParams.cbegin();
        while (true) {
            if (it->isConstant)
                os << it->constant;
            else
                os << "%" << it->variable;
            if (++it != gateParams.cend()) {
                os << ",";
                continue;
            }
            break;
        }
        os << ")";
    }

    // target qubits
    os << " ";
    printVector(os, qubits, " ");
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

std::ostream& ParameterDefStmt::print(std::ostream& os) const {
    os << "#" << refNumber << " = { ";
    assert(gateMatrix.isParametrizedMatrix());

    auto it = gateMatrix.pData().cbegin();
    it->print(os);
    while (++it != gateMatrix.pData().cend())
        it->print(os << ", ");
    return os << " }\n";
}

QuantumGate QuantumCircuit::gateApplyToQuantumGate(const GateApplyStmt& gateApplyStmt) {
    if (gateApplyStmt.paramRefNumber >= 0) {
        for (auto it = paramDefs.begin(); it != paramDefs.end(); it++) {
            if (it->refNumber == gateApplyStmt.paramRefNumber)
                return QuantumGate(it->gateMatrix, gateApplyStmt.qubits);
        }
        assert(false && "Cannot find parameter def stmt");
        return QuantumGate();
    }
    return QuantumGate(GateMatrix::FromParameters(
                gateApplyStmt.name, gateApplyStmt.gateParams),
            gateApplyStmt.qubits);
}

CircuitGraph QuantumCircuit::toCircuitGraph() {
    CircuitGraph graph;
    for (const auto& s : stmts) {
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
