#include "saot/CostModel.h"

using namespace saot;

int main() {
  PerformanceCache cache;
  CPUKernelGenConfig cpuConfig;
  cache.runExperiments(cpuConfig, 20, 1);
  return 0;
}