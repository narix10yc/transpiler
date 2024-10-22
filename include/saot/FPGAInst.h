#ifndef SAOT_FPGAINST_H
#define SAOT_FPGAINST_H

#include <vector>
#include <iostream>
#include <cassert>

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

    MemoryInst() : op(MOp_NUL), qIdx(-1), cycle(-1) {}

    MemoryInst(MemoryOp op, int qIdx = -1, int cycle = -1)
        : op(op), qIdx(qIdx), cycle(cycle) {}
    
    bool isNull() const { return op == MOp_NUL; }

    std::ostream& print(std::ostream&) const;

};

class GateInst {
public:
    GateOp op;
    int gateID;
    std::vector<int> qubits;

    GateInst() : op(GOp_NUL), gateID(-1), qubits() {}

    GateInst(GateOp op, int gateID, std::initializer_list<int> qubits = {})
            : op(op), gateID(gateID), qubits(qubits) {
        assert(op == GOp_NUL ^ gateID >= 0);
    }

    GateInst(GateOp op, int gateID, const std::vector<int>& qubits = {})
            : op(op), gateID(gateID), qubits(qubits) {
        assert(op == GOp_NUL ^ gateID >= 0);
    }

    bool isNull() const { return op == GOp_NUL; }

    std::ostream& print(std::ostream&) const;
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
    
};

struct FPGAInstGenConfig {
public:
    int gridSize;

static const FPGAInstGenConfig Grid2x2;
static const FPGAInstGenConfig Grid3x3;
static const FPGAInstGenConfig Grid4x4;
};

// top-level function to generate FPGA instructions from a CircuitGraph
std::vector<Instruction> genInstruction(
        const CircuitGraph&, const FPGAInstGenConfig&);

}; // namespace saot::fpga

#endif // SAOT_FPGAINST_H