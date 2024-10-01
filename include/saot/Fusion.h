#ifndef SAOT_FUSION_H
#define SAOT_FUSION_H

#include <iostream>
#include <cassert>

namespace saot {

class CircuitGraph;

class FusionConfig {
public:
    int maxNQubits;
    int maxOpCount;
    double zeroSkippingThreshold;
    bool allowMultipleTraverse;
    bool incrementScheme;
    static FusionConfig Disable;
    static FusionConfig TwoQubitOnly;
    static FusionConfig Default;
    static FusionConfig Aggressive;

    static FusionConfig Preset(int level) {
        if (level == 0)
            return FusionConfig::Disable;
        if (level == 1) 
            return FusionConfig::TwoQubitOnly;
        if (level == 2) 
            return FusionConfig::Default;
        if (level == 3)
            return FusionConfig::Aggressive;
        assert(false && "Unsupported FusionConfig preset");
        return FusionConfig::Default;
    }

    std::ostream& display(std::ostream&) const;
};

void applyGateFusion(const FusionConfig&, CircuitGraph&);

} // namespace saot

#endif // SAOT_FUSION_H