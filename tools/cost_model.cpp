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

static auto&
ArgOutputFilename = cl::registerArgument<std::string>("o")
  .desc("Output file name")
  .setOccurExactlyOnce();

static auto&
ArgPrecision = cl::registerArgument<int>("precision")
  .desc("Specify precision (f32 or f64)")
  .setParser(parsePrecision).setOccurAtMostOnce().init(-1);

static auto&
ArgF32 = cl::registerArgument<bool>("f32")
  .desc("Use single-precision")
  .init(false);

static auto&
ArgF64 = cl::registerArgument<bool>("f64")
  .desc("Use double-precision")
  .init(false);

// return true on error
static bool checkPrecisionArgsCollision() {
  if (ArgF32 && ArgF64)
    return true;
  if (ArgF32) {
    if (ArgPrecision == 64)
      return true;
    ArgPrecision.init(32);
    return false;
  }
  if (ArgF64) {
    if (ArgPrecision == 32)
      return true;
    ArgPrecision.init(64);
    return false;
  }
  if (ArgPrecision != 32 && ArgPrecision != 64)
    return true;
  return false;
}

static auto&
ArgNQubits = cl::registerArgument<int>("nqubits")
  .desc("Specify number of qubits")
  .init(26).setOccurAtMostOnce();

static auto&
ArgNThreads = cl::registerArgument<int>("T")
  .desc("Specify number of threads")
  .setArgumentPrefix().setOccurExactlyOnce();

static auto&
ArgOverwriteMode = cl::registerArgument<bool>("overwrite")
  .desc("Overwrite the output file with new results")
  .init(false);

using namespace cast;

int main(int argc, char** argv) {
  cl::ParseCommandLineArguments(argc, argv);

  if (checkPrecisionArgsCollision()) {
    std::cerr << BOLDRED("[Error]: ")
              << "Precision arguments contradict with each other.\n";
    return 1;
  }

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

