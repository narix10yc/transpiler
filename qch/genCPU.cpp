#include "qch/ast.h"
#include "simulation/cpu.h"
#include "simulation/types.h"

using namespace qch::ast;
using namespace simulation;

void CircuitStmt::genCPU(CPUGenContext& ctx) const {
    ctx.nqubits = nqubits;
    for (auto& s : stmts)
        s->genCPU(ctx);
}

void GateApplyStmt::genCPU(CPUGenContext& ctx) const {
    uint64_t idxMax = 1ULL << (ctx.nqubits - ctx.vecSizeInBits - 1);
    if (name != "u3") {
        std::cerr << "skipped gate " << name << "\n";
        return;
    }

    auto mat = OptionalComplexMatrix2::FromEulerAngles(parameters[0], parameters[1], parameters[2]);

    auto u3 = ir::U3Gate { static_cast<uint8_t>(qubits[0]), mat.ToIRMatrix(1e-8) };
                                
    auto func = ctx.getGenerator().genU3(u3);
    std::string funcName = func->getName().str();

    ctx.declStream << "void " << funcName;
    if (ctx.getRealTy() == ir::RealTy::Double)
        ctx.declStream << "(double*, double*, uint64_t, uint64_t, v8double);\n";
    else
        ctx.declStream << "(float*, float*, uint64_t, uint64_t, v8float);\n";

    ctx.kernelStream << "  " << funcName << "(real, imag, 0, " << idxMax << ",\n    "
        << ((ctx.getRealTy() == ir::RealTy::Double) ? "(v8double){" : "(v8float){")
        << std::setprecision(16)
        << mat.ar.value_or(0) << "," << mat.br.value_or(0) << ","
        << mat.cr.value_or(0) << "," << mat.dr.value_or(0) << ","
        << mat.ai.value_or(0) << "," << mat.bi.value_or(0) << ","
        << mat.ci.value_or(0) << "," << mat.di.value_or(0) << "});\n";

    ctx.gateCount ++;
}

