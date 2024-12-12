#include "saot/FPGAInst.h"
#include "saot/QuantumGate.h"

using namespace saot;
using namespace saot::fpga;

namespace {
inline bool isRealOnlyGate(const QuantumGate& gate, double reTol) {
  const auto* cMat = gate.gateMatrix.getConstantMatrix();
  assert(cMat);
  for (const auto& cplx : *cMat) {
    if (std::abs(cplx.imag()) > reTol)
      return false;
  }
  return true;
}
} // namespace

FPGAGateCategory
saot::fpga::getFPGAGateCategory(const QuantumGate& gate,
                                const FPGAGateCategoryTolerance &tolerances) {
  switch (gate.gateMatrix.gateKind) {
  case gX:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp |
           FPGAGateCategory::RealOnly;
  case gY:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp;
  case gZ:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp;
  case gP:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::UnitaryPerm;
  case gH:
    return FPGAGateCategory::SingleQubit;
  case gCX:
    return FPGAGateCategory::NonComp;
  case gCZ:
    return FPGAGateCategory::NonComp;
  case gCP:
    return FPGAGateCategory::UnitaryPerm;
  default:
    break;
  }

  FPGAGateCategory cate = FPGAGateCategory::General;

  if (gate.qubits.size() == 1)
    cate |= FPGAGateCategory::SingleQubit;

  if (const auto* p = gate.gateMatrix.getUnitaryPermMatrix(tolerances.upTol)) {
    if (std::all_of(
            p->data.begin(), p->data.end(),
            [tol = tolerances.ncTol](const std::pair<size_t, double>& d) {
              std::complex<double> cplx(std::cos(d.second), std::sin(d.second));
              return (std::abs(cplx - std::complex<double>(1.0, 0.0)) <= tol) ||
                     (std::abs(cplx - std::complex<double>(-1.0, 0.0)) <=
                      tol) ||
                     (std::abs(cplx - std::complex<double>(0.0, 1.0)) <= tol) ||
                     (std::abs(cplx - std::complex<double>(0.0, -1.0)) <= tol);
            })) {
      cate |= FPGAGateCategory::NonComp;
    } else
      cate = FPGAGateCategory::UnitaryPerm;
  }

  if (isRealOnlyGate(gate, tolerances.reOnlyTol))
    cate |= FPGAGateCategory::RealOnly;

  return cate;
}
