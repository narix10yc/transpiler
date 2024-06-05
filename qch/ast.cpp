#include "qch/ast.h"
#include <cmath>

using namespace qch::ast;

void GateApplyStmt::print(std::ostream& os) const {
    os << "  " << name;
    auto pSize = parameters.size();

    // parameter
    if (pSize > 0) {
        os << "(";
        for (size_t i = 0; i < pSize-1; i++)
            os << parameters[i] << ",";
        os << parameters[pSize-1] << ")";
    }

    // qubits
    os << " ";
    auto qSize = qubits.size();
    for (size_t i = 0; i < qSize-1; i++)
        os << qubits[i] << " ";

    os << qubits[qSize-1] << "\n";
}

void CircuitStmt::print(std::ostream& os) const {
    os << "circuit<" << nqubits << "> " << name << "()\n{\n";
    
    for (auto& s : stmts)
        s->print(os);

    os << "}\n";
}
