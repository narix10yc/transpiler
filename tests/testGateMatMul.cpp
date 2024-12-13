#include "saot/QuantumGate.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

using namespace saot;
using namespace saot::test;
using namespace utils;

template<unsigned simd_s>
static void internal() {
  TestSuite suite("MatMul between Gates (s = " + std::to_string(simd_s) + ")");

  constexpr int nqubits = 6;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nqubits - 1);
  for (int repeat = 0; repeat < 5; ++repeat) {
    QuantumGate gate0(utils::randomUnitaryMatrix(2), d(gen));
    QuantumGate gate1(utils::randomUnitaryMatrix(2), d(gen));
    auto gate = gate1.lmatmul(gate0);

    utils::StatevectorAlt<double, simd_s> sv0(nqubits), sv1(nqubits);
    sv0.randomize();
    sv1 = sv0;

    sv0.applyGate(gate0).applyGate(gate1);
    sv1.applyGate(gate);

    std::stringstream ss;
    ss << "Apply U gate on qubits " << gate0.qubits[0] << " and " << gate1.qubits[0];
    suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, ss.str(), GET_INFO());
  }
  suite.displayResult();
}

void saot::test::test_gateMatMul() {
  internal<1>();
  internal<2>();
}