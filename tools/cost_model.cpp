#include "cast/CostModel.h"
#include "utils/CommandLine.h"

namespace cl = utils::cl;

static bool parsePrecision(utils::StringRef clValue, int& valueToWriteOn) {
  if (clValue.compare("32") == 0 || clValue.compare_ci("single") == 0 || 
      clValue.compare("F32") == 0 || clValue.compare("f32") == 0) {
    valueToWriteOn = 32;
    return false;
  }
  if (clValue.compare("64") == 0 || clValue.compare_ci("double") == 0 || 
      clValue.compare("F64") == 0 || clValue.compare("f64") == 0) {
    valueToWriteOn = 64;
    return false;
  }
  return true;
}

// static auto&
// ArgOutputFilename = cl::registerArgument<std::string>("o");

static auto&
ArgPrecision = cl::registerArgument<int>("precision")
  .setParser(parsePrecision).setValueFormat(cl::VF_Required);

static auto&
ArgF32 = cl::registerArgument<bool>("f32");

using namespace cast;

int main(int argc, char** argv) {
  cl::ParseCommandLineArguments(argc, argv);
  cl::DisplayArguments();

  // PerformanceCache cache;
  // CPUKernelGenConfig cpuConfig;
  // cpuConfig.simd_s = 1;
  // cache.runExperiments(cpuConfig, 28, 10, 100);
  // cache.saveToCSV("threads10");

  // cache = PerformanceCache::LoadFromCSV("threads10.csv");
  // std::cerr << cache.items.size() << " items found!\n";

  // StandardCostModel costModel(&cache);
  // costModel.display(std::cerr);

  // auto gate = QuantumGate::RandomUnitary(2, 3, 4, 5, 6);

  // std::cerr << "OpCount = " << gate.opCount(1e-8) << "\n";

  // costModel.computeSpeed(gate, 32, 10);
  return 0;
}

