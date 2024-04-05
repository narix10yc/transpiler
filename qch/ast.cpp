#include "qch/ast.h"

using namespace qch::ast;

void GateApply::print(std::ostream& os) const {
    os << name;
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
    for (size_t i = 0; i < qSize; i++)
        os << qubits[i] << " ";

    os << qubits[qSize] << "\n";
}