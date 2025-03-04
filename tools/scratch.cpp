#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
#include "simulation/KernelManager.h"

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
    cudaKernelMgr.emitPTX(1);
  }, "PTX Code Emission");

  // llvm::errs() << cudaKernelMgr.kernels()[0].ptxString << "\n";

  cudaKernelMgr.initCUJIT(2, /* verbose */ 1);

  return 0;
}