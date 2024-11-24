#include "saot/CostModel.h"
#include "saot/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>

using namespace saot;

int StandardCostModel::getCost(const QuantumGate &gate) const {
  assert(0 && "Not Implemented");
  return -1;
}

int AdaptiveCostModel::getCost(const QuantumGate &gate) const {
  assert(0 && "Not Implemented");
  return -1;
}

void PerformanceCache::saveToCSV(const std::string &_fileName) const {
  std::string fileName = _fileName;
  auto l = fileName.size();
  if (l >= 4 && fileName.substr(l - 4, l) != ".csv")
    fileName += ".csv";

  std::ofstream file(fileName);
  assert(file.is_open());

  file << "nqubits,op_count,time\n";
  for (const auto &entry : entries) {
    file << entry.nqubits << "," << entry.opCount << "," << std::scientific
         << std::setw(6) << entry.memUpdateSpeed << "\n";
  }
  file.close();
}

PerformanceCache PerformanceCache::LoadFromCSV(const std::string &fileName) {
  assert(0 && "Not Implemented");
  return PerformanceCache();
}

void PerformanceCache::addExperiments(int comprehensiveness) {}