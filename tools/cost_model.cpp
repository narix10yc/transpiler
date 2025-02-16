#include "cast/CostModel.h"

using namespace cast;

int main() {
  PerformanceCache cache;
  CPUKernelGenConfig cpuConfig;
  cache.runExperiments(cpuConfig, 24, 10, 50);
  cache.saveToCSV("threads10");
  //
  // cache = PerformanceCache::LoadFromCSV("threads10.csv");
  // std::cerr << cache.items.size() << " items found!\n";
  //
  // StandardCostModel costModel(&cache);
  // costModel.display(std::cerr);
  //
  // auto gate = QuantumGate::RandomUnitary(2, 3, 4, 5, 6);
  //
  // std::cerr << "OpCount = " << gate.opCount(1e-8) << "\n";
  //
  // costModel.computeSpeed(gate, 32, 10);
  return 0;
}

