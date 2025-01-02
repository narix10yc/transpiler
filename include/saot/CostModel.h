#ifndef SAOT_COSTMODEL_H
#define SAOT_COSTMODEL_H

#include "simulation/KernelGen.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraphContext.h"
#include <cassert>
#include <string>
#include <vector>

namespace saot {

struct CostResult {
  double benefit;
  QuantumGate* fusedGate;
};

class PerformanceCache;

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const {
    assert(false && "Should not call from base class");
    return { 0.0, nullptr };
  }
};

class NaiveCostModel : public CostModel {
  int maxNQubits;
  int maxOp;
  double zeroTol;

public:
  NaiveCostModel(int maxNQubits, int maxOp, double zeroTol)
    : maxNQubits(maxNQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};

class StandardCostModel : public CostModel {
  PerformanceCache* cache;

  struct UpdateSpeedCollection {
    int nThreads;
    int precision;
    int nData; // number of data points
    double totalMemSpd;
  };
  std::vector<UpdateSpeedCollection> updateSpeeds;
public:
  StandardCostModel(PerformanceCache* cache);

  double computeExpectedMemSpd(const QuantumGate& gate) const;

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};

class AdaptiveCostModel : public CostModel {
public:
  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};



class PerformanceCache {
public:
  struct Item {
    int nqubits;
    int opCount;
    int precision;
    /// This is approximately how many shuffling operations are needed in each
    /// amplitude loading process, calculated by 1 << (number of loBits)
    int irregularity;
    int nThreads;
    double memUpdateSpeed;
  };

  std::vector<Item> items;
  PerformanceCache() : items() {}

  void runExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nqubits, int nThreads, int comprehensiveness);

  void saveToCSV(const std::string& fileName) const;
  
  static PerformanceCache LoadFromCSV(const std::string& fileName);
};

} // namespace saot

#endif // SAOT_COSTMODEL_H