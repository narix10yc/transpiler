#include "openqasm/ast.h"
#include "saot/CircuitGraph.h"

using namespace openqasm::ast;

saot::CircuitGraph RootNode::toCircuitGraph() const {
    saot::CircuitGraph graph;
    std::vector<int> qubits;
    for (const auto& s : stmts) {
        saot::GateMatrix::gate_params_t params;
        int i = 0;
        auto gateApply = dynamic_cast<GateApplyStmt*>(s.get());
        if (gateApply == nullptr) {
            std::cerr << "skipping " << s->toString() << "\n";
            continue;
        }
        qubits.clear();
        for (const auto& t : gateApply->targets) {
            qubits.push_back(static_cast<unsigned>(t->getIndex()));
        }
        for (const auto& p : gateApply->parameters) {
            auto ev = p->getExprValue();
            assert(ev.isConstant);
            params[i++] = ev.value;
        }
        auto matrix = saot::GateMatrix::FromName(gateApply->name, params);
        graph.addGate(matrix, qubits);
    }

    return graph;
}