#include "simulation/KernelManager.h"

using namespace cast;

int main() {

  KernelManager kernelMgr;
  GPUKernelGenConfig gpuGenConfig;
  gpuGenConfig.displayInfo(std::cerr) << "\n";


  kernelMgr.genGPUKernel(
    gpuGenConfig, QuantumGate::RandomUnitary(4, 6), "my_gpu_kernel");


  return 0;
}