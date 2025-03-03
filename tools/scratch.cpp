#include "simulation/KernelManager.h"

using namespace cast;

int main() {

  CUDAKernelManager cudaKernelMgr;
  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.displayInfo(std::cerr) << "\n";

  cudaKernelMgr.genCUDAKernel(
    cudaGenConfig,
    std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(4, 6)),
    "my_cuda_kernel");

  cudaKernelMgr.emitPTX(1);

  llvm::errs() << cudaKernelMgr.kernels()[0].ptxString << "\n";

  return 0;
}