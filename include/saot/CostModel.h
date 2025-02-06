#ifndef SAOT_COSTMODEL_H
#define SAOT_COSTMODEL_H

#include "simulation/KernelManager.h"
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

  virtual double computeSpeed(
      const QuantumGate& gate, int precision, int nThreads) const = 0;

  virtual CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const {
    assert(false && "Should not call from base class");
    return { 0.0, nullptr };
  }
};

class NaiveCostModel : public CostModel {
  int maxnQubits;
  int maxOp;
  double zeroTol;

public:
  NaiveCostModel(int maxnQubits, int maxOp, double zeroTol)
    : maxnQubits(maxnQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  double computeSpeed(
      const QuantumGate& gate, int precision, int nThreads) const override;

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};

/// \c StandardCostModel assumes simulation time is proportional to opCount and
/// independent to target qubits.
class StandardCostModel : public CostModel {
  PerformanceCache* cache;
  double zeroTol;

  /// Collect memory update speeds for quick loading and lookup
  struct UpdateSpeedCollection {
    int nQubits;
    int precision;
    int nThreads;
    int nData; // number of data points;
    double totalTimePerOpCount;

    double getMemSpd(int opCount) const {
      return static_cast<double>(nData) / (totalTimePerOpCount * opCount);
    }
  };
  std::vector<UpdateSpeedCollection> updateSpeeds;
public:
  StandardCostModel(PerformanceCache* cache, double zeroTol = 1e-8);

  std::ostream& display(std::ostream& os, int nLines = 0) const;

  double computeSpeed(
    const QuantumGate& gate, int precision, int nThreads) const override;

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};

class AdaptiveCostModel : public CostModel {
public:
  double computeSpeed(
      const QuantumGate &gate, int precision, int nThreads) const override {
    assert(false && "Not Implemented");
    return 0.0;
  }

  CostResult computeBenefit(
      const QuantumGate& lhsGate, const QuantumGate& rhsGate,
      CircuitGraphContext& context) const override;
};



class PerformanceCache {
public:
  struct Item {
    int nQubits;
    int opCount;
    int precision;
    /// This is approximately how many shuffling operations are needed in each
    /// amplitude loading process, calculated by 1 << (number of loBits)
    /// TODO: Not in use yet
    int irregularity;
    int nThreads;
    double memUpdateSpeed;
  };

  std::vector<Item> items;
  PerformanceCache() : items() {}

  void runExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits, int nThreads, int comprehensiveness);

  void saveToCSV(const std::string& fileName) const;
  
  static PerformanceCache LoadFromCSV(const std::string& fileName);
};

} // namespace saot

#endif // SAOT_COSTMODEL_H