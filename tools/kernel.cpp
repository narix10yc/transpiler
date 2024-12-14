#include "simulation/KernelGen.h"
#include "saot/QuantumGate.h"
#include "utils/square_matrix.h"

#include "llvm/Support/CommandLine.h"

using namespace saot;

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::LLVMContext llvmContext;
  llvm::Module llvmModule("myModule", llvmContext);

  CPUKernelGenConfig config {
    .simd_s = 1,
    // .forceDenseKernel = true,
    .matrixLoadMode = CPUKernelGenConfig::UseMatImmValues,
  };

  QuantumGate gate(GateMatrix(utils::randomUnitaryMatrix(4)), {0, 1});

  auto* llvmFunc = genCPUCode(llvmModule, config, gate, "kernel_");
  // llvmFunc->print(llvm::errs(), nullptr);

  return 0;
}