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

    auto u3 = U3Gate::FromAngles(qubits[0],
        parameters[0], parameters[1], parameters[2]);

    std::stringstream funcNameSS;
    funcNameSS << "u3_" << ctx.gateCount << "_" << std::hex << u3.getID();

    std::string funcName = funcNameSS.str();

    ctx.getGenerator().genU3(u3, funcName);

    ctx.incStream << "void " << funcName
        << "(double*, double*, int64_t, int64_t, v8double);\n";

    ctx.cStream << "  " << funcName << "(real, imag, 0, " << idxMax << ",\n    "
        << "(v8double){"
        << u3.mat.ar.value_or(0) << "," << u3.mat.br.value_or(0) << ","
        << u3.mat.cr.value_or(0) << "," << u3.mat.dr.value_or(0) << ","
        << u3.mat.ai.value_or(0) << "," << u3.mat.bi.value_or(0) << ","
        << u3.mat.ci.value_or(0) << "," << u3.mat.di.value_or(0) << "});\n";

    ctx.gateCount ++;
}

