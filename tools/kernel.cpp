#include "simulation/KernelGen.h"
#include "saot/QuantumGate.h"

using namespace saot;


int main(int argc, char** argv) {
  llvm::LLVMContext llvmContext;
  llvm::Module llvmModule("myModule", llvmContext);

  CPUKernelGenConfig config {
    .simdS = 3,
    .forceDenseKernel = true,
  };

  assert(argc > 1);
  QuantumGate gate(GateMatrix::MatrixH_c, std::stoi(argv[1]));
  for (int i = 2; i < argc; i++)
    gate = gate.lmatmul(QuantumGate(GateMatrix::MatrixH_c, std::stoi(argv[i])));

  auto* llvmFunc = saot::genCPUCode(llvmModule, config, gate, "kernel_");
  // llvmFunc->print(llvm::errs(), nullptr);
  // llvmFunc->getBasicBlockList().
  
  // llvmModule.print(llvm::errs(), nullptr);

  return 0;
}