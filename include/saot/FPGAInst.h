#ifndef SAOT_FPGAINST_H
#define SAOT_FPGAINST_H

#include <vector>
#include <iostream>
#include <cassert>

namespace saot {
    class CircuitGraph;
    class QuantumGate;
    class GateBlock;
} // namespace saot

namespace saot::fpga {

enum FPGAGateCategory : unsigned {
    fpgaGeneral = 0,
    fpgaSingleQubit = 0b0001,
    
    // unitary permutation
    fpgaUnitaryPerm = 0b0010,
    
    // Non-computational is a special sub-class of unitary permutation where all
    // non-zero entries are +1, -1, +i, -i.
    fpgaNonComp = 0b0110,
    fpgaRealOnly = 0b1000,
    
    // composite
    fpgaSingleQubitUnitaryPerm = 0b0011,
    fpgaSingleQubitNonComp = 0b0111,
};

FPGAGateCategory getFPGAGateCategory(const QuantumGate& gate);

enum GateOp : int {
    GOp_NUL = 0,

    GOp_SQ = 1,     // Single Qubit
    GOp_UP = 2,     // Unitary Permutation
};

enum MemoryOp : int {
    MOp_NUL = 0,
    MOp_SSR = 1,    // Shuffle Swap Row
    MOp_SSC = 2,    // Shuffle Swap Col

    MOp_FSR = 3,    // Full Swap Row
    MOp_FSC = 4,    // Full Swap Col

    MOp_EXT = 5,    // External Memory Swap
};

class MemoryInst {
public:
    MemoryOp op;
    int qIdx;
    int cycle;

    MemoryInst() : op(MOp_NUL), qIdx(-1), cycle(-1) {}

    MemoryInst(MemoryOp op, int qIdx = -1, int cycle = -1)
        : op(op), qIdx(qIdx), cycle(cycle) {}
    
    bool isNull() const { return op == MOp_NUL; }

    std::ostream& print(std::ostream&) const;

};

class GateInst {
public:
    GateOp op;
    GateBlock* block;

    GateInst() : op(GOp_NUL), block(nullptr) {}

    GateInst(GateOp op, GateBlock* block)
            : op(op), block(block) {
        assert((op == GOp_NUL) ^ (block != nullptr));
    }

    bool isNull() const { return op == GOp_NUL; }

    std::ostream& print(std::ostream&) const;
};


struct FPGACostConfig {
    // external memory swap
    double tExtMemOp;
    // memOp with no gateOp
    double tMemOpOnly;
    // single-qubit gateOp with real gate
    double tRealGate;
    // unitary-perm gateOp
    double tUnitaryPerm;
    // single-qubit gateOp
    double tGeneral;

    FPGACostConfig(double tMemOpOnly, double tRealGate,
                   double tUnitaryPerm, double tGeneral)
        : tMemOpOnly(tMemOpOnly),
          tRealGate(tRealGate),
          tUnitaryPerm(tUnitaryPerm),
          tGeneral(tGeneral) {}
};


class Instruction {
public:
    MemoryInst memInst;
    GateInst gateInst;

    Instruction() : memInst(), gateInst() {}

    Instruction(const MemoryInst& memInst, const GateInst& gateInst)
        : memInst(memInst), gateInst(gateInst) {}

    std::ostream& print(std::ostream& os) const {
        memInst.print(os) << " : ";
        gateInst.print(os) << "\n";
        return os;
    }

    bool isNull() const {
        return memInst.isNull() && gateInst.isNull();
    }

    uint64_t cost(const FPGACostConfig&) const;
    
};

struct FPGAInstGenConfig {
public:
    int gridSize;
    int nOnChipQubits;

    FPGAInstGenConfig(int gridSize, int nOnChipQubits) 
        : gridSize(gridSize), nOnChipQubits(nOnChipQubits) {}
};

// top-level function to generate FPGA instructions from a CircuitGraph
std::vector<Instruction> genInstruction(
        const CircuitGraph&, const FPGAInstGenConfig&);

}; // namespace saot::fpga

#endif // SAOT_FPGAINST_H