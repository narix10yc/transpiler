#ifndef CAST_COSTMODEL_H
#define CAST_COSTMODEL_H

#include "simulation/KernelManager.h"
#include "cast/QuantumGate.h"
#include "cast/CircuitGraphContext.h"
#include <cassert>
#include <string>
#include <vector>

namespace cast {

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
};

/// @brief \c NaiveCostModel is based on the size and operation count of fused
/// gates.
class NaiveCostModel : public CostModel {
  int maxNQubits;
  int maxOp;
  double zeroTol;

public:
  NaiveCostModel(int maxNQubits, int maxOp, double zeroTol)
    : maxNQubits(maxNQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  double computeSpeed(
      const QuantumGate& gate, int precision, int nThreads) const override;
};

/// \c StandardCostModel assumes simulation time is proportional to opCount and
/// independent to target qubits.
class StandardCostModel : public CostModel {
  PerformanceCache* cache;
  double zeroTol;
  double maxMemUpdateSpd;

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

    double getMemSpd(int opCount, double cap) const {
      double spd = static_cast<double>(nData) / (totalTimePerOpCount * opCount);
      return std::min(spd, cap);
    }
  };
  std::vector<UpdateSpeedCollection> updateSpeeds;
public:
  StandardCostModel(PerformanceCache* cache, double zeroTol = 1e-8);

  std::ostream& display(std::ostream& os, int nLines = 0) const;

  double computeSpeed(
    const QuantumGate& gate, int precision, int nThreads) const override;
};

class AdaptiveCostModel : public CostModel {
public:
  double computeSpeed(
      const QuantumGate &gate, int precision, int nThreads) const override {
    assert(false && "Not Implemented");
    return 0.0;
  }
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
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;
  };

  std::vector<Item> items;
  PerformanceCache() : items() {}

  void runExperiments(
      const CPUKernelGenConfig& cpuConfig,
      int nQubits, int nThreads, int nRuns);

  void writeResults(std::ostream& os) const;
  
  static PerformanceCache LoadFromCSV(const std::string& fileName);
  
  constexpr static const char*
  CSV_Title = "nQubits,opCount,precision,irregularity,nThreads,memSpd\n";
};


} // namespace cast

#endif // CAST_COSTMODEL_H