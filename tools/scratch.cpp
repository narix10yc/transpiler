#include "utils/square_matrix.h"
#include "utils/statevector.h"
#include "utils/utils.h"
#include "saot/QuantumGate.h"

using namespace saot;
using namespace utils;

int main() {
  StatevectorAlt<double, 1> sv(4);
  QuantumGate gate(GateMatrix(randomUnitaryMatrix(2)), 1);
  sv.applyGate(gate);

  return 0;
}