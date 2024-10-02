#ifndef SAOT_FUSION_H
#define SAOT_FUSION_H

#include <iostream>
#include <cassert>

namespace saot {

class CircuitGraph;

class CPUFusionConfig {
public:
    int maxNQubits;
    int maxOpCount;
    double zeroSkippingThreshold;
    bool allowMultipleTraverse;
    bool incrementScheme;
    static CPUFusionConfig Disable;
    static CPUFusionConfig TwoQubitOnly;
    static CPUFusionConfig Default;
    static CPUFusionConfig Aggressive;

    static CPUFusionConfig FpgaCanonicalForm;

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

} // namespace saot

#endif // SAOT_FUSION_H