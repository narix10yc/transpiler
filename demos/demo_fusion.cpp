#include "simulation/StatevectorCPU.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "openqasm/parser.h"

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;

using namespace cast;

cl::opt<std::string>
ArgInputFilename("i",
  cl::desc("Input file name"), cl::Positional, cl::Required);

cl::opt<std::string>
ArgModelPath("model",
  cl::desc("Path to performance model"), cl::Required);

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::Required);

cl::opt<int>
ArgSimd_s("simd-s", cl::desc("simd s"), cl::init(1));

cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::desc("Run no-fuse circuit"), cl::init(false));

int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  openqasm::Parser qasmParser(ArgInputFilename, 0);
  auto qasmRoot = qasmParser.parse();

  // This is temporary work-around as CircuitGraph does not allow copy yet
  CircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  qasmRoot->toCircuitGraph(graphNoFuse);
  qasmRoot->toCircuitGraph(graphNaiveFuse);
  qasmRoot->toCircuitGraph(graphAdaptiveFuse);

  CPUFusionConfig fusionConfig = CPUFusionConfig::Aggressive;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = ArgNThreads;
  // NaiveCostModel costModel(3, 0, 1e-8);

  NaiveCostModel naiveCostModel(5, -1, 1e-8);
  applyCPUGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuse);

  auto cache = PerformanceCache::LoadFromCSV(ArgModelPath);
  StandardCostModel standardCostModel(&cache);
  applyCPUGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuse);

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = ArgSimd_s;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  // Generate kernels
  if (ArgRunNoFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromCircuitGraph(
        kernelGenConfig, graphNoFuse, "graphNoFuse");
    }, "Generate No-fuse Kernels");
  }
  utils::timedExecute([&]() {
    kernelMgr.genCPUGatesFromCircuitGraph(
      kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
  }, "Generate Naive-fused Kernels");
  utils::timedExecute([&]() {
    kernelMgr.genCPUGatesFromCircuitGraph(
      kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
  }, "Generate Adaptive-fused Kernels");


  // JIT compile kernels
  std::vector<CPUKernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;
  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      ArgNThreads,
      llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false,
      /* verbose */ 1);
    double opCountTotal = 0.0;
    if (ArgRunNoFuse) {
      opCountTotal = 0.0;
      kernelsNoFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphNoFuse");
      for (const auto* kernel : kernelsNoFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "No-fuse total opCount: " << opCountTotal << "\n";
    }
    opCountTotal = 0.0;
    kernelsNaiveFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphNaiveFuse");
    for (const auto* kernel : kernelsNaiveFuse)
      opCountTotal += kernel->gate->opCount(1e-8);
    std::cerr << "Naive-fuse total opCount: " << opCountTotal << "\n";

    opCountTotal = 0.0;
    kernelAdaptiveFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphAdaptiveFuse");
    for (const auto* kernel : kernelAdaptiveFuse)
      opCountTotal += kernel->gate->opCount(1e-8);
    std::cerr << "Adaptive-fuse total opCount: " << opCountTotal << "\n";
  }, "JIT compile kernels");

  // Run kernels
  utils::StatevectorCPU<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();
  timeit::Timer timer(/* replication */ 1);
  timeit::TimingResult tr;

  if (ArgRunNoFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNoFuse)
        kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, ArgNThreads);
    });
    tr.display(3, std::cerr << "No-fuse Circuit:\n");
  }

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelsNaiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, ArgNThreads);
  });
  tr.display(3, std::cerr << "Naive-fused Circuit:\n");

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelAdaptiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, ArgNThreads);
  });
  tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");

  return 0;
}