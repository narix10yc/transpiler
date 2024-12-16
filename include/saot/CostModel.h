#ifndef SAOT_COSTMODEL_H
#define SAOT_COSTMODEL_H

#include "simulation/KernelGen.h"
#include <cassert>
#include <string>
#include <vector>

namespace saot {

class QuantumGate;

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual int getCost(const QuantumGate& gate) const {
    assert(false && "Should not call from base class");
    return -1;
  }
};

class StandardCostModel : public CostModel {
  int maxNQubits;
  int maxOp;

public:
  StandardCostModel(int maxNQubits, int maxOp)
      : maxNQubits(maxNQubits), maxOp(maxOp) {}

  int getCost(const QuantumGate& gate) const override;
};

class AdaptiveCostModel : public CostModel {
public:
  int getCost(const QuantumGate& gate) const override;
};

class PerformanceCache {
public:
  struct Item {
    int nqubits;
    int opCount;
    int nThreads;
    double memUpdateSpeed;
  };

  std::vector<Item> items;
  PerformanceCache() : items() {}

  void runExperiments(
    const CPUKernelGenConfig& cpuConfig, int nqubits, int comprehensiveness);

  void saveToCSV(const std::string& fileName) const;
  
  static PerformanceCache LoadFromCSV(const std::string& fileName);
};

} // namespace saot

#endif // SAOT_COSTMODEL_H