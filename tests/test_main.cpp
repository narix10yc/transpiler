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

int main() {
  testH</* simdS */ 1>();
  testH</* simdS */ 2>();

  return 0;
}