#include "openqasm/ast.h"
#include "saot/CircuitGraph.h"

using namespace openqasm::ast;

saot::CircuitGraph RootNode::toCircuitGraph() const {
    saot::CircuitGraph graph;
    std::vector<int> qubits;
    std::vector<double> params;
    for (const auto& s : stmts) {
        auto gateApply = dynamic_cast<GateApplyStmt*>(s.get());
        if (gateApply == nullptr) {
            std::cerr << "skipping " << s->toString() << "\n";
            continue;
        }
        qubits.clear();
        for (const auto& t : gateApply->targets) {
            qubits.push_back(static_cast<unsigned>(t->getIndex()));
        }
        params.clear();
        for (const auto& p : gateApply->parameters) {
            auto ev = p->getExprValue();
            assert(ev.isConstant);
            params.push_back(ev.value);
        }
        auto matrix = saot::GateMatrix::FromName(gateApply->name, params);
        graph.addGate(matrix, qubits);
    }

    return graph;
}