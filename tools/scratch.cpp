#include "simulation/KernelManager.h"
#include "simulation/StatevectorCUDA.h"
#include "cast/CircuitGraph.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <span>

using namespace cast;

std::shared_ptr<QuantumGate> getH(int q) {
  return std::make_shared<QuantumGate>(QuantumGate::H(q));
}

int main() {
  CUDAKernelManager kernelMgrCUDA;
  CUDAKernelGenConfig genConfigCUDA;
  kernelMgrCUDA.genCUDAGate(genConfigCUDA, getH(0), "H0");
  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, 1);
  kernelMgrCUDA.initCUJIT(2, 1);

  utils::StatevectorCUDA<double> svCUDA(6);
  svCUDA.initialize();
  svCUDA.sync();

  std::cerr << kernelMgrCUDA.getPTXString(0) << "\n";

  utils::printArray(std::cerr, std::span(svCUDA.hData(), svCUDA.size())) << "\n";
  kernelMgrCUDA.launchCUDAKernel(svCUDA.dData(), svCUDA.nQubits(), 0);
  svCUDA.sync();
  utils::printArray(std::cerr, std::span(svCUDA.hData(), svCUDA.size())) << "\n";

  // svCUDA.randomize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;
  // cudaDeviceSynchronize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;

  

  return 0;
}