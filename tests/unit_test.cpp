#include "tests/Test.h"
#include "utils/statevector.h"
#include "saot/QuantumGate.h"
#include "simulation/KernelGen.h"
#include "simulation/JIT.h"
#include "utils/iocolor.h"

using namespace saot;
using namespace saot::test;
using namespace utils::statevector;

using namespace llvm;

int main() {
  TestResultRegistry registry;

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  CPUKernelGenConfig cpuConfig;

  QuantumGate gate(GateMatrix::MatrixH_c, 0);

  auto* llvmFunc = genCPUCode(*llvmModule, cpuConfig, gate, "gate_h_q0");

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  auto expectedFunc = jit->lookup(llvmFunc->getName());
  if (!expectedFunc) {
    errs() << RED("Error in finding function") << "\n";
    return 1;
  }
  auto func = expectedFunc->toPtr<void(void*, uint64_t, uint64_t, void*)>();
  StatevectorAlt<double, 1> sv(/* nqubits */ 2, /* initialize */ true);
  sv.print() << "\n";

  func(sv.data, 0ULL, 1ULL, nullptr);
  sv.print() << "\n";

  registry.displayResult();

  return 0;
}