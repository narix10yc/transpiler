#include "tests/TestKit.h"
#include "utils/statevector.h"
#include "saot/QuantumGate.h"
#include "simulation/KernelGen.h"
#include "simulation/JIT.h"
#include "utils/iocolor.h"

using namespace saot;
using namespace saot::test;
using namespace utils::statevector;

using namespace llvm;

#define FUNC_TYPE void(void*, uint64_t, uint64_t, void*)

// testH
#include "test_h.inc"
// testApplyGate
#include "test_applyGate.inc"

/// @brief Test general single-qubit unitary gates
// template<unsigned simd_s>
// void test_U() {
//   TestSuite suite("Gate U with simd_s = " + std::to_string(simd_s));
//
//   StatevectorAlt<double, simd_s> sv(5);
//   sv.initialize();
//   for (int q = 0; q < sv.nqubits; q++)
//     sv.applyGate(QuantumGate(GateMatrix::MatrixH_c, q));
//   for (int q = 0; q < sv.nqubits; q++) {
//     suite.assertClose(sv.prob(q), 0.5,
//       GET_INFO("Apply round H: Prob at qubit " + std::to_string(q)));
//   }
//
//   sv.randomize();
//   suite.assertClose(sv.norm(), 1.0, GET_INFO("Rand SV: Norm"));
//
//   // phase gates do not change probabilities
//   for (int q = 0; q < sv.nqubits; q++) {
//     double pBefore = sv.prob(q);
//     auto gate0 = QuantumGate(GateMatrix::FromName("p", {0.14}), q);
//     auto gate1 = QuantumGate(GateMatrix::FromName("p", {0.41}), (q+1) % sv.nqubits);
//     auto gate = gate0.lmatmul(gate1);
//
//     sv.applyGate(gate);
//     std::stringstream ss;
//     ss << "Phase gate at qubits "
//        << q << " " << ((q+1) % sv.nqubits);
//     suite.assertClose(sv.norm(), 1.0, GET_INFO(ss.str() + ": Norm"));
//     suite.assertClose(pBefore, sv.prob(q), GET_INFO(ss.str() + ": Prob"));
//   }
//
//   suite.displayResult();
// }

int main() {
  testH</* simd_s */ 1>();
  testH</* simd_s */ 2>();

  test_applyGate<1>();
  test_applyGate<2>();

  return 0;
}