#ifndef SAOT_COSTMODEL_H
#define SAOT_COSTMODEL_H

#include "simulation/KernelGen.h"
#include "saot/QuantumGate.h"
#include <cassert>
#include <string>
#include <vector>

namespace saot {

struct CostResult {
  double benefit;
  std::unique_ptr<QuantumGate> fusedGate;
};

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual CostResult computeBenefit(const QuantumGate& lhsGate,
                                    const QuantumGate& rhsGate) const {
    assert(false && "Should not call from base class");
    return { 0.0, nullptr };
  }
};

class StandardCostModel : public CostModel {
  int maxNQubits;
  int maxOp;
  double zeroTol;

public:
  StandardCostModel(int maxNQubits, int maxOp, double zeroTol)
    : maxNQubits(maxNQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate) const override;
};

class AdaptiveCostModel : public CostModel {
public:
  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate) const override;
};

class PerformanceCache {
public:
  struct Item {
    int nqubits;
    int opCount;
    /// This is approximately how many shuffling operations are needed in each
    /// amplitude loading process, calculated by 1 << (number of loBits)
    int irregularity;
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