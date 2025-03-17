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

std::shared_ptr<QuantumGate> getU(int q) {
  return std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(q));
}

int main() {
  CUDAKernelManager kernelMgrCUDA;
  CUDAKernelGenConfig genConfigCUDA;
  genConfigCUDA.precision = 32;
  kernelMgrCUDA.genCUDAGate(genConfigCUDA, getU(0), "H0");
  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, 0);
  kernelMgrCUDA.initCUJIT(2, 0);

  utils::StatevectorCUDA<float> svCUDA(6);
  svCUDA.initialize();
  // svCUDA.randomize();
  svCUDA.sync();

  std::cerr << kernelMgrCUDA.getPTXString(0) << "\n";

  utils::printArray(std::cerr, std::span(svCUDA.hData(), svCUDA.size())) << "\n";
  kernelMgrCUDA.launchCUDAKernel(svCUDA.dData(), svCUDA.nQubits(), 0);

  svCUDA.sync();
  utils::printArray(std::cerr, std::span(svCUDA.hData(), svCUDA.size())) << "\n";

  std::cerr << "Norm: " << svCUDA.norm() << "\n";
  // cudaDeviceSynchronize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;

  

  return 0;
}