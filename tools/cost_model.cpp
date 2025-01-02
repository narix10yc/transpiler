#include "saot/CostModel.h"

using namespace saot;

int main() {
  PerformanceCache cache;
  // CPUKernelGenConfig cpuConfig;
  // cache.runExperiments(cpuConfig, 24, 10, 1);
  // cache.saveToCSV("threads10");

  cache = PerformanceCache::LoadFromCSV("threads10.csv");
  std::cerr << cache.items.size() << " items found!\n";
  return 0;
}