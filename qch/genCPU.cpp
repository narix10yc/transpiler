#include "qch/ast.h"

using namespace qch::ast;

void GateApplyStmt::genCPU(simulation::CPUGenContext& ctx) const {
    if (name != "u3") {
        std::cerr << "skipped gate " << name << "\n";
        return;
    }



}

