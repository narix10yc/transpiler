#include <utils/statevector.h>

#include "saot/Parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "openqasm/parser.h"

#define N_THREADS 10

using namespace saot;

int main(int argc, const char** argv) {
  assert(argc > 1);

  openqasm::Parser qasmParser(argv[1], 0);
  auto qasmRoot = qasmParser.parse();
  // Parser parser("../examples/rqc/q6_67_52.qasm");
  // auto qc = parser.parseQuantumCircuit();
  // qc.print(std::cerr);

  // This is temporary work-around as CircuitGraph does not allow copy yet
  CircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  qasmRoot->toCircuitGraph(graphNoFuse);
  qasmRoot->toCircuitGraph(graphNaiveFuse);
  qasmRoot->toCircuitGraph(graphAdaptiveFuse);

  CPUFusionConfig fusionConfig = CPUFusionConfig::Default;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = 10;
  // NaiveCostModel costModel(3, 0, 1e-8);

  auto cache = PerformanceCache::LoadFromCSV("threads10.csv") ;
  StandardCostModel standardCostModel(&cache);
  applyCPUGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuse);

  NaiveCostModel naiveCostModel(5, -1, 1e-8);
  applyCPUGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuse);

  KernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  kernelMgr.genCPUFromGraph(kernelGenConfig, graphNoFuse, "graphNoFuse");
  kernelMgr.genCPUFromGraph(kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
  kernelMgr.genCPUFromGraph(kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
  std::vector<KernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;
  utils::timedExecute([&]() {
    kernelMgr.initJIT(N_THREADS, llvm::OptimizationLevel::O1);
    kernelsNoFuse = kernelMgr.collectCPUGraphKernels("graphNoFuse");
    kernelsNaiveFuse = kernelMgr.collectCPUGraphKernels("graphNaiveFuse");
    kernelAdaptiveFuse = kernelMgr.collectCPUGraphKernels("graphAdaptiveFuse");
  }, "JIT compile kernels");

  utils::StatevectorAlt<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();
  utils::timedExecute([&]() {
    for (auto* kernel : kernelsNoFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  }, "Simulation on the no-fuse circuit");

  utils::timedExecute([&]() {
  for (auto* kernel : kernelsNaiveFuse)
    kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  }, "Simulation on the naive-fused circuit");

  utils::timedExecute([&]() {
  for (auto* kernel : kernelAdaptiveFuse)
    kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  }, "Simulation on the adaptive-fused circuit");

  return 0;
}