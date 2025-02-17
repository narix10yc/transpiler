#include "utils/statevector.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "openqasm/parser.h"

#define N_THREADS 10

using namespace cast;

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
  fusionConfig.nThreads = N_THREADS;
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

  utils::timedExecute([&]() {
    kernelMgr.genCPUFromGraph(kernelGenConfig, graphNoFuse, "graphNoFuse");
  }, "Generate No-fuse Kernels");
  utils::timedExecute([&]() {
    kernelMgr.genCPUFromGraph(kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
  }, "Generate Naive-fused Kernels");
  utils::timedExecute([&]() {
    kernelMgr.genCPUFromGraph(kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
  }, "Generate Adaptive-fused Kernels");

  std::vector<KernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;
  utils::timedExecute([&]() {
    kernelMgr.initJIT(N_THREADS, llvm::OptimizationLevel::O1, /* verbose */ 1);
    kernelsNoFuse = kernelMgr.collectCPUGraphKernels("graphNoFuse");
    kernelsNaiveFuse = kernelMgr.collectCPUGraphKernels("graphNaiveFuse");
    kernelAdaptiveFuse = kernelMgr.collectCPUGraphKernels("graphAdaptiveFuse");
  }, "JIT compile kernels");

  utils::StatevectorAlt<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();

  timeit::Timer timer(/* replication */ 5);
  timeit::TimingResult tr;

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelsNoFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  });
  tr.display(3, std::cerr << "No-fuse Circuit:\n");

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelsNaiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  });
  tr.display(3, std::cerr << "Naive-fused Circuit:\n");

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelAdaptiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data, sv.nQubits, *kernel, N_THREADS);
  });
  tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");

  return 0;
}