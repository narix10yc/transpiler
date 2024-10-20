#ifndef SAOT_FPGAINST_H
#define SAOT_FPGAINST_H

#include <vector>
#include <iostream>

namespace saot {
    class CircuitGraph;
}

namespace saot::fpga {


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
};

class MemoryInst {
public:
    MemoryOp op;
    int qIdx;
    int cycle;

    MemoryInst(MemoryOp op, int qIdx = -1, int cycle = -1)
        : op(op), qIdx(qIdx), cycle(cycle) {}

    std::ostream& print(std::ostream&);

};

class GateInst {
public:
    GateOp op;
    int arg;

    GateInst(GateOp op, int arg = -1) : op(op), arg(arg) {}

    std::ostream& print(std::ostream&);
};

class Instruction {
public:
    MemoryInst memInst;
    GateInst gateInst;

    Instruction(const MemoryInst& memInst, const GateInst& gateInst)
        : gateInst(gateInst), memInst(memInst) {}

    std::ostream& print(std::ostream& os) {
        memInst.print(os) << " : ";
        gateInst.print(os) << "\n";
        return os;
    }

    bool isNull() const {
        return memInst.op == MOp_NUL && gateInst.op == GOp_NUL;
    }
    
};

struct FPGAInstGenConfig {
public:
    int gridSize;

static const FPGAInstGenConfig Default;
};

// top-level function to generate FPGA instructions from a CircuitGraph
std::vector<Instruction> genInstruction(
        const CircuitGraph&, const FPGAInstGenConfig&);

}; // namespace saot::fpga

#endif // SAOT_FPGAINST_H