#ifndef SAOT_FUSION_H
#define SAOT_FUSION_H

#include <iostream>
#include <cassert>

#include "saot/FPGAConfig.h"

namespace saot {

class CircuitGraph;
class QuantumGate;

struct CPUFusionConfig {
    int maxNQubits;
    int maxOpCount;
    double zeroSkippingThreshold;
    bool allowMultipleTraverse;
    bool incrementScheme;
    static CPUFusionConfig Disable;
    static CPUFusionConfig TwoQubitOnly;
    static CPUFusionConfig Default;
    static CPUFusionConfig Aggressive;

    static CPUFusionConfig Preset(int level) {
        if (level == 0)
            return CPUFusionConfig::Disable;
        if (level == 1) 
            return CPUFusionConfig::TwoQubitOnly;
        if (level == 2) 
            return CPUFusionConfig::Default;
        if (level == 3)
            return CPUFusionConfig::Aggressive;
        assert(false && "Unsupported CPUFusionConfig preset");
        return CPUFusionConfig::Default;
    }

    std::ostream& display(std::ostream&) const;
};

void applyCPUGateFusion(const CPUFusionConfig&, CircuitGraph&);

void applyFPGAGateFusion(CircuitGraph&, const FPGAFusionConfig&);

} // namespace saot

#endif // SAOT_FUSION_H