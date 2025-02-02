#ifndef SAOT_CPUFUSION_H
#define SAOT_CPUFUSION_H

#include "saot/CostModel.h"

#include <cassert>
#include <iostream>

#include "saot/FPGAConfig.h"

namespace saot {

class CircuitGraph;
class QuantumGate;

struct CPUFusionConfig {
  int precision;
  int nThreads;

  double zeroTol;
  bool multiTraverse;
  bool incrementScheme;
  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin;

  static CPUFusionConfig Preset(int level) {
    if (level == 0)
      return Disable;
    if (level == 1)
      return Minor;
    if (level == 2)
      return Default;
    if (level == 3)
      return Aggressive;
    assert(false && "Unknown CPUFusionConfig Preset");
    return Disable;
  }

  static const CPUFusionConfig Disable;
  static const CPUFusionConfig Minor;
  static const CPUFusionConfig Default;
  static const CPUFusionConfig Aggressive;

  std::ostream& display(std::ostream&) const;
};

void applyCPUGateFusion(
    const CPUFusionConfig&, const CostModel*, CircuitGraph&);

void applyFPGAGateFusion(CircuitGraph&, const FPGAFusionConfig&);

} // namespace saot

#endif // SAOT_CPUFUSION_H