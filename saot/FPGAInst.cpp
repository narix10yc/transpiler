#include "saot/FPGAInst.h"
#include "saot/QuantumGate.h"

using namespace saot;
using namespace saot::fpga;

namespace {
bool isRealOnlyGate(const QuantumGate& gate) {
    const auto* cMat = gate.gateMatrix.getConstantMatrix();
    assert(cMat);
    for (const auto& cplx : cMat->data) {
        if (cplx.imag() != 0.0)
            return false;
    }
    return true;
}
} // anynomous namespace

FPGAGateCategory saot::fpga::getFPGAGateCategory(const QuantumGate& gate) {
    FPGAGateCategory cate;
    switch (gate.gateMatrix.gateKind){
        case gX: cate = fpgaSingleQubitNonComp; break;
        case gY: cate = fpgaSingleQubitNonComp; break;
        case gZ: cate = fpgaSingleQubitNonComp; break;
        case gP: cate = fpgaSingleQubitUnitaryPerm; break;
        case gH: cate = fpgaSingleQubit; break;
        case gCX: cate = fpgaNonComp; break;
        case gCZ: cate = fpgaNonComp; break;
        case gCP: cate = fpgaUnitaryPerm; break;
        default: {
            if (const auto* p = gate.gateMatrix.getUnitaryPermMatrix()) {
                cate = fpgaUnitaryPerm;
                break;
            }
            cate = fpgaGeneral;
            break;
        }
    }

    if (isRealOnlyGate(gate))
        cate = static_cast<FPGAGateCategory>(cate | fpgaRealOnly);

    // TODO: handle general gates
    return cate;
}
