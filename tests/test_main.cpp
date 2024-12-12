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
// testU
#include "test_u.inc"

template<unsigned simd_s>
void test_gateMatMul() {
  TestSuite suite("MatMul between Gates");

  constexpr int nqubits = 6;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nqubits - 1);
  for (int repeat = 0; repeat < 5; ++repeat) {
    QuantumGate gate0(utils::randomUnitaryMatrix(2), d(gen));
    QuantumGate gate1(utils::randomUnitaryMatrix(2), d(gen));
    auto gate = gate1.lmatmul(gate0);

    utils::statevector::StatevectorAlt<double, simd_s> sv0(nqubits), sv1(nqubits);
    sv0.randomize();
    sv1 = sv0;

    sv0.applyGate(gate0).applyGate(gate1);
    sv1.applyGate(gate);

    std::stringstream ss;
    ss << "Apply U gate on qubits " << gate0.qubits[0] << " and " << gate1.qubits[0];
    suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, GET_INFO(ss.str()));
  }
  suite.displayResult();
}

int main() {
  test_H</* simd_s */ 1>();
  test_H</* simd_s */ 2>();

  test_applyGate<1>();
  test_applyGate<2>();

  test_gateMatMul<1>();

  test_U<1>();
  test_U<2>();

  return 0;
}