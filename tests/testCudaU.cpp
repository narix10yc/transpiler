#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"
#include <random>

using namespace cast;
using namespace cast::test;

template<unsigned nQubits>
static void f() {
  test::TestSuite suite(
    "Gate U1q (" + std::to_string(nQubits) + " qubits)");
  utils::StatevectorCPU<double> svCPU(nQubits, /* simd_s */ 0);
  utils::StatevectorCUDA<double> svCUDA0(nQubits), svCUDA1(nQubits);

  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0;
    cudaMemcpy(svCPU.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
      cudaMemcpyDeviceToHost);
  };

  CUDAKernelManager kernelMgrCUDA;

  // generate random unitary gates
  std::vector<std::shared_ptr<QuantumGate>> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    gates.emplace_back(
      std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(q)));
  }

  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;
  for (int q = 0; q < nQubits; q++) {
    kernelMgrCUDA.genCUDAGate(
      cudaGenConfig, gates[q], "gateImm_" + std::to_string(q));
  }

  // cudaGenConfig.forceDenseKernel = true;
  // cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;
  // for (int q = 0; q < nQubits; q++) {
  //   kernelMgrCUDA.genCUDAGate(
  //     cudaGenConfig, gates[q], "gateConstMemSpace_" + std::to_string(q));
  // }

  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, /* verbose */ 1);
  kernelMgrCUDA.initCUJIT(2, /* verbose */ 1);
  for (unsigned i = 0; i < nQubits; i++) {
    randomizeSV();
    std::stringstream ss;
    ss << "Apply U1q at " << gates[i]->qubits[0];
    // auto immFuncName = "gateImm_" + std::to_string(i);
    // auto loadFuncName = "gateConstMemSpace_" + std::to_string(i);
    kernelMgrCUDA.launchCUDAKernel(svCUDA0.dData(), svCUDA0.nQubits(), i);
    // kernelMgr.launchCUDAKernel(sv1.data, sv1.nQubits, i);
    svCPU.applyGate(*gates[i]);
    suite.assertClose(svCUDA0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    // suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    // suite.assertClose(utils::fidelity(sv0, sv2), 1.0,
    //   ss.str() + ": Imm Fidelity", GET_INFO());
    // suite.assertClose(utils::fidelity(sv1, sv2), 1.0,
    //   ss.str() + ": Load Fidelity", GET_INFO());
  }
  suite.displayResult();
}

void test::test_cudaU() {
  f<8>();
  // f<12>();
}
