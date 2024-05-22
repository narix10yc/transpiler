#include "qch/ast.h"
#include "simulation/cpu.h"
#include "simulation/types.h"

using namespace qch::ast;
using namespace simulation;

void CircuitStmt::genCPU(CPUGenContext& ctx) const {
    for (auto& s : stmts)
        s->genCPU(ctx);
}

void GateApplyStmt::genCPU(CPUGenContext& ctx) const {
    // type string
    std::string typeStr =
        (ctx.getRealTy() == ir::RealTy::Double) ? "double*" : "float*";
    if (name == "u3") {
        // get matrix
        auto mat = OptionalComplexMatrix2::FromEulerAngles(
                                    parameters[0], parameters[1], parameters[2]);
        auto u3 = ir::U3Gate { static_cast<uint8_t>(qubits[0]), mat.ToIRMatrix(1e-8) };
        uint32_t u3ID = u3.getID();

        // get function name
        std::string funcName;
        if (auto it = ctx.u3GateMap.find(u3ID); it != ctx.u3GateMap.end()) {
            funcName = it->second;
        } else {
            // generate gate kernel
            auto func = ctx.getGenerator().genU3(u3);
            funcName = func->getName().str();
            ctx.u3GateMap[u3ID] = funcName;
        }
        
        ctx.u3Params.push_back(mat.toArray());
        ctx.kernels.push_back({funcName, 1});
    } else if (name == "u2q") {
        ComplexMatrix4 mat;
        for (size_t i = 0; i < 16; i++) {
            mat.real[i] = parameters[2*i];
            mat.imag[i] = parameters[2*i + 1];
        }

        U2qGate u2q { static_cast<uint8_t>(qubits[0]),
                      static_cast<uint8_t>(qubits[1]), mat };

        ir::U2qGate u2qIR = u2q.ToIRGate();
        
        auto u2qRepr = u2qIR.getRepr();
        std::string funcName;
        if (auto it = ctx.u2qGateMap.find(u2qRepr); it != ctx.u2qGateMap.end()) {
            funcName = it->second;
        } else {
            auto func = ctx.getGenerator().genU2q(u2qIR);
            funcName = func->getName().str();
            ctx.u2qGateMap[u2qRepr] = funcName;
        }

        std::array<double, 32> arr;
        for (size_t i = 0; i < 16; i++) {
            arr[i] = mat.real[i];
            arr[i+16] = mat.imag[i];
        }

        ctx.u2qParams.push_back(arr);
        ctx.kernels.push_back({funcName, 2});
    } else {
        std::cerr << "unrecognized gate " << name << " to CPU gen\n";
        return;
    }

}

