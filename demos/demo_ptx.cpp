#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
#include "simulation/KernelManager.h"

#include "timeit/timeit.h"
#include "simulation/StatevectorCUDA.h"

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1);
  Parser parser(argv[1]);
  auto qc = parser.parseQuantumCircuit();
  CircuitGraph graph;
  qc.toCircuitGraph(graph);

  CUDAKernelManager cudaKernelMgr;
  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.displayInfo(std::cerr) << "\n";

  utils::timedExecute([&]() {
    cudaKernelMgr.genCUDAGatesFromCircuitGraph(cudaGenConfig, graph, "myGraph");
  }, "CUDA Kernel Generation");

  utils::timedExecute([&]() {
    cudaKernelMgr.emitPTX(2, llvm::OptimizationLevel::O1, /* verbose */ 1);
  }, "PTX Code Emission");

  utils::timedExecute([&]() {
    cudaKernelMgr.initCUJIT(3, /* verbose */ 1);
  }, "CUDA JIT Initialization");

  utils::StatevectorCUDA<double> sv(28);
  sv.initialize();

  timeit::Timer timer;
  auto tr = timer.timeit([&]() {
    cudaKernelMgr.launchCUDAKernel(sv.dData, sv._nQubits, 0);
    cuCtxSynchronize();
  });

  tr.display();

  return 0;
}