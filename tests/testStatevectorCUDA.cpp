#include "tests/TestKit.h"
#include "simulation/StatevectorCUDA.h"

using namespace cast::test;

template<int nQubits>
static void f() {
  cast::test::TestSuite suite(
    "StatevectorCUDA with " + std::to_string(nQubits) + " qubits");
  utils::StatevectorCUDA<float> svF32(nQubits);
  utils::StatevectorCUDA<double> svF64(nQubits);
  
  svF32.initialize();
  svF64.initialize();
  suite.assertClose(svF32.norm(), 1.0f, "initialize norm F32", GET_INFO());
  suite.assertClose(svF64.norm(), 1.0, "initialize norm F64", GET_INFO());

  svF32.randomize();
  svF64.randomize();
  suite.assertClose(svF32.norm(), 1.0f, "randomize norm F32", GET_INFO());
  suite.assertClose(svF64.norm(), 1.0, "randomize norm F64", GET_INFO());

  suite.displayResult();
}

void cast::test::test_statevectorCUDA() {
  f<8>();
  f<12>();
  f<16>();
}
