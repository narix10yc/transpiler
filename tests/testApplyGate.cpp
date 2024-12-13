#include "saot/QuantumGate.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

using namespace saot;
using namespace utils;

template<unsigned simd_s>
static void internal() {
  test::TestSuite suite("applyGate (s = " + std::to_string(simd_s) + ")");

  StatevectorAlt<double, simd_s> sv(5);
  sv.initialize();
  for (int q = 0; q < sv.nqubits; q++)
    sv.applyGate(QuantumGate(GateMatrix::MatrixH_c, q));
  for (int q = 0; q < sv.nqubits; q++) {
    suite.assertClose(sv.prob(q), 0.5,
      "Apply round H: Prob at qubit " + std::to_string(q), GET_INFO());
  }

  sv.randomize();
  suite.assertClose(sv.norm(), 1.0, "Rand SV: Norm", GET_INFO());

  // phase gates do not change probabilities
  for (int q = 0; q < sv.nqubits; q++) {
    double pBefore = sv.prob(q);
    auto gate0 = QuantumGate(GateMatrix::FromName("p", {0.14}), q);
    auto gate1 = QuantumGate(GateMatrix::FromName("p", {0.41}), (q+1) % sv.nqubits);
    auto gate = gate0.lmatmul(gate1);

    sv.applyGate(gate);
    std::stringstream ss;
    ss << "Phase gate at qubits "
       << q << " " << ((q+1) % sv.nqubits);
    suite.assertClose(sv.norm(), 1.0, ss.str() + ": Norm", GET_INFO());
    suite.assertClose(pBefore, sv.prob(q), ss.str() + ": Prob", GET_INFO());
  }

  suite.displayResult();
}

void test::test_applyGate() {
  internal<1>();
  internal<2>();
}