#include "cast/CostModel.h"
#include "utils/CommandLine.h"

#include <fstream>

#define ERR_PRECISION 1
#define ERR_FILENAME 2
#define ERR_FILE_IO 3

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
ArgForceFilename = cl::registerArgument<bool>("force")
  .desc("Force output filename as it is (possibly not ending with '.csv')")
  .init(false);

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
  if (ArgF32 && ArgF64) {
    std::cerr << BOLDRED("[Error]: ")
              << "-f32 and -f64 cannot be set together.\n";
    return true;
  }
  if (ArgF32) {
    if (ArgPrecision == 64) {
      std::cerr << BOLDRED("[Error]: ")
                << "Precision arguments contradict with each other.\n";
      return true;
    }
    ArgPrecision.init(32);
    return false;
  }
  if (ArgF64) {
    if (ArgPrecision == 32) {
      std::cerr << BOLDRED("[Error]: ")
                << "Precision arguments contradict with each other.\n";
      return true;
    }
    ArgPrecision.init(64);
    return false;
  }
  if (ArgPrecision != 32 && ArgPrecision != 64) {
    std::cerr << BOLDRED("[Error]: ")
              << "Precision should be either 32 or 64.\n";
    return true;
  }
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

static auto&
ArgSimdS = cl::registerArgument<int>("simd-s")
  .desc("simd_s").init(1);

static auto&
ArgNTests = cl::registerArgument<int>("N")
  .desc("Specify number of tests to perform")
  .setArgumentPrefix().setOccurExactlyOnce();

using namespace cast;

bool checkFileName() {
  if (ArgForceFilename)
    return false;

  const std::string& fileName = ArgOutputFilename;
  if (fileName.length() > 4 && fileName.ends_with(".csv"))
    return false;

  std::cerr << BOLDYELLOW("Notice: ")
            << "Output filename does not end with '.csv'. "
               "If this filename is desired, please add '-force' "
               "commandline option\n";
  return true;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineArguments(argc, argv);
  if (checkPrecisionArgsCollision())
    return ERR_PRECISION;
  
  cl::DisplayArguments();
  
  if (checkFileName())
    return ERR_FILENAME;

  std::ifstream inFile;
  std::ofstream outFile;
  if (ArgOverwriteMode)
    outFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    outFile.open(ArgOutputFilename, std::ios::app);

  inFile.open(ArgOutputFilename, std::ios::in);
  if (!outFile || !inFile) {
    std::cerr << BOLDRED("[Error]: ")
              << "Unable to open file '" << ArgOutputFilename << "'.\n";
    return ERR_FILE_IO;
  }

  if (inFile.peek() == std::ifstream::traits_type::eof())
    outFile << "nQubits,opCount,precision,irregularity,nThreads,memSpd\n";
  inFile.close();

  PerformanceCache cache;
  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = ArgSimdS;
  cache.runExperiments(cpuConfig, ArgNQubits, ArgNThreads, ArgNTests);
  cache.writeResults(outFile);

  outFile.close();
  return 0;
}

