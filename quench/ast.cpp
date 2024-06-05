#include "quench/ast.h"

using namespace quench::ast;

std::ostream& RootNode::print(std::ostream& os) const {
    for (const auto& s : stmts)
        s->print(os);
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



void CircuitStmt::addGate(std::unique_ptr<GateApplyStmt> gate) {

}