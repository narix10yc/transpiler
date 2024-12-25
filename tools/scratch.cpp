#include "saot/QuantumGate.h"

using namespace saot;

int main() {
  auto gate = QuantumGate::RandomUnitary<2>({1, 3});
  std::cerr << gate.opCount(0.0) << "\n";

  return 0;
}