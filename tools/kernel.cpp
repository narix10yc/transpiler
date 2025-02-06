#include <utils/statevector.h>

#include "saot/Parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "openqasm/parser.h"

using namespace saot;

int main(int argc, const char** argv) {
  openqasm::Parser qasmParser("../examples/rqc/q8_112_76.qasm", 0);
  auto qasmRoot = qasmParser.parse();
  // Parser parser("../examples/rqc/q6_67_52.qasm");
  // auto qc = parser.parseQuantumCircuit();
  // qc.print(std::cerr);

  CircuitGraph graph;
  // qc.toCircuitGraph(graph);
  qasmRoot->toCircuitGraph(graph);
  graph.print(std::cerr << "Before Fusion:\n", 2) << "\n";


  CPUFusionConfig fusionConfig = CPUFusionConfig::Default;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = 10;
  // NaiveCostModel costModel(3, 0, 1e-8);

  auto cache = PerformanceCache::LoadFromCSV("threads10.csv") ;
  StandardCostModel costModel(&cache);
  costModel.display(std::cerr);

  applyCPUGateFusion(fusionConfig, &costModel, graph);
  graph.print(std::cerr << "After Fusion:\n", 2) << "\n";

  // auto fusedQC = ast::QuantumCircuit::FromCircuitGraph(graph);
  // fusedQC.print(std::cerr << "Fused QuantumCircuit Serialization\n") << "\n";

  KernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  kernelMgr.genCPUFromGraph(kernelGenConfig, graph, "myGraph");
  kernelMgr.initJIT();
  auto kernels = kernelMgr.collectCPUGraphKernels("myGraph");
  std::cerr << kernels.size() << " kernel found\n";

  utils::StatevectorAlt<double> sv(graph.nQubits, kernelGenConfig.simd_s);
  sv.randomize();
  for (const auto* kernel : kernels) {
    kernelMgr.applyCPUKernel(sv.data, sv.nQubits, *kernel);
  }

  return 0;
}