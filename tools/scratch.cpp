#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "cast/CostModel.h"
#include "simulation/KernelManager.h"

#include "utils/statevector.h"

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1);
  Parser parser(argv[1]);
  auto qc = parser.parseQuantumCircuit();

  CircuitGraph graph;
  qc.toCircuitGraph(graph);

  graph.print(std::cerr);

  CPUFusionConfig fusionConfig;
  NaiveCostModel costModel(3, 999999, 1e-8);

  applyCPUGateFusion(fusionConfig, &costModel, graph, 3);
  
  graph.print(std::cerr << "\nCircuitGraph after Fusion:\n");


  KernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;
  kernelMgr.genCPUFromGraph(kernelGenConfig, graph, "myGraph");

  kernelMgr.initJIT(10, llvm::OptimizationLevel::O1, false, 1);

  utils::StatevectorAlt<double> sv(graph.nQubits, kernelGenConfig.simd_s, false);
  sv.randomize();

  auto kernels = kernelMgr.collectCPUGraphKernels("myGraph");
  for (auto* kernel : kernels) {
    kernelMgr.applyCPUKernel(sv.data, sv.nQubits, *kernel);
  }
  

  




  
  return 0;
}