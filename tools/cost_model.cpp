#include "saot/CostModel.h"

using namespace saot;

int main() {
  PerformanceCache cache;
  CPUKernelGenConfig cpuConfig;
  cache.runExperiments(cpuConfig, 10, 1);
  cache.saveToCSV("tmp");
  return 0;
}