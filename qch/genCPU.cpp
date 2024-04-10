#include "qch/ast.h"
#include "simulation/cpu.h"
#include "simulation/types.h"

using namespace qch::ast;
using namespace simulation;

void GateApplyStmt::genCPU(simulation::CPUGenContext& ctx) const {
    int64_t idxMax = 1 << (ctx.nqubits - ctx.vecSizeInBits);
    if (name != "u3") {
        std::cerr << "skipped gate " << name << "\n";
        return;
    }

    // auto mat = 
    // OptionalComplexMat2x2::FromAngles(parameters[0], parameters[1], parameters[2]);
    // U3Gate gate { mat, qubits[0] };
    // auto id = gate.getID();

    std::string funcName = "gate_u3_" + std::to_string(ctx.gateCount);

    ctx.getGenerator().genU3(qubits[0], funcName, 
                parameters[0], parameters[1], parameters[2]);

    ctx.incStream << "void " << funcName
        << "(double*, double*, int64_t, int64_t, double, double, double);\n";

    ctx.cStream << funcName << "(real, real, 0, " << idxMax << ", "
        << parameters[0] << ", " << parameters[1] << ", " << parameters[2]
        << ");\n";

    ctx.gateCount ++;
}

