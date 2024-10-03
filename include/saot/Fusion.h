#ifndef SAOT_FUSION_H
#define SAOT_FUSION_H

#include <iostream>
#include <cassert>

namespace saot {

class CircuitGraph;

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

enum FPGAGateCategory : unsigned {
    fpgaGeneral = 0,
    fpgaSingleQubit = 0b001,
    fpgaUnitaryPerm = 0b010, // unitary permutation
    // Non-computational is a special sub-class of unitary permutation where all
    // non-zero entries are +1, -1, +i, -i.
    fpgaNonComp = 0b110,
    
    // composite
    fpgaSingleQubitUnitaryPerm = 0b011,
    fpgaSingleQubitNonComp = 0b111,
};

struct FPGAFusionConfig {
    int maxUnitaryPermutationSize;
    bool ignoreSingleQubitNonCompGates;

    static FPGAFusionConfig Default;
};

void applyFPGAGateFusion(const FPGAFusionConfig&, CircuitGraph&);

} // namespace saot

#endif // SAOT_FUSION_H