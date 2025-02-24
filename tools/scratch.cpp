#include "simulation/KernelManager.h"

using namespace cast;

int main() {

  KernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 3;

  kernelMgr.genCPUKernel(kernelGenConfig, QuantumGate::RandomUnitary(4, 6), "myKernel");

  

  return 0;
}