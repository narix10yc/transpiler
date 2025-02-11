#include <utils/statevector.h>

#include "saot/Parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "openqasm/parser.h"

using namespace saot;

int main(int argc, const char** argv) {
  assert(argc > 1);

  openqasm::Parser qasmParser(argv[1], 0);
  auto qasmRoot = qasmParser.parse();
  // Parser parser("../examples/rqc/q6_67_52.qasm");
  // auto qc = parser.parseQuantumCircuit();
  // qc.print(std::cerr);

  CircuitGraph graph;
  // qc.toCircuitGraph(graph);
  qasmRoot->toCircuitGraph(graph);
  // graph.print(std::cerr << "Before Fusion:\n", 2) << "\n";


  CPUFusionConfig fusionConfig = CPUFusionConfig::Default;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = 10;
  // NaiveCostModel costModel(3, 0, 1e-8);

  // auto cache = PerformanceCache::LoadFromCSV("threads10.csv") ;
  // StandardCostModel costModel(&cache);
  // costModel.display(std::cerr);

  NaiveCostModel costModel(5, -1, 1e-8);

  applyCPUGateFusion(fusionConfig, &costModel, graph);
  // graph.print(std::cerr << "After Fusion:\n", 2) << "\n";

  KernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  kernelMgr.genCPUFromGraph(kernelGenConfig, graph, "myGraph");
  std::vector<KernelInfo*> kernels;
  utils::timedExecute([&]() {
    kernelMgr.initJIT(llvm::OptimizationLevel::O1);
    kernels = kernelMgr.collectCPUGraphKernels("myGraph");
  }, "JIT compile kernels");
  std::cerr << kernels.size() << " kernel found\n";

  utils::StatevectorAlt<double> sv(graph.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();
  utils::timedExecute([&]() {
    for (const auto* kernel : kernels) {
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, 10);
    }
  }, "Simulation on the fused circuit");

  utils::timedExecute([&]() {
    for (const auto* kernel : kernels) {
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, 10);
    }
  }, "Simulation on the fused circuit");

  return 0;
}