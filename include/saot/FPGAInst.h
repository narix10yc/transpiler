#ifndef SAOT_FPGAINST_H
#define SAOT_FPGAINST_H

#include <memory>
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

enum GInstKind : int {
    GOp_NUL = 0,

    GOp_SQ = 1,     // Single Qubit
    GOp_UP = 2,     // Unitary Permutation
};

enum MInstKind : int {
    MOp_NUL = 0,
    MOp_SSR = 1,    // Shuffle Swap Row
    MOp_SSC = 2,    // Shuffle Swap Col

    MOp_FSR = 3,    // Full Swap Row
    MOp_FSC = 4,    // Full Swap Col

    MOp_EXT = 5,    // External Memory Swap
};


class MemoryInst {
private:
    MInstKind mKind;
public:
    MemoryInst(MInstKind mKind) : mKind(mKind) {}

    virtual ~MemoryInst() = default;

    MInstKind getKind() const { return mKind; }

    bool isNull() { return getKind() == MOp_NUL; }
    virtual std::ostream& print(std::ostream& os) const {
        assert(false && "Calling from base class");
        return os;
    }
};

// Null (NUL)
class MInstNUL : public MemoryInst {
public:
    MInstNUL() : MemoryInst(MOp_NUL) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "NUL" << std::string(12, ' ');
    }
};

// Shuffle Swap Row (SSR)
class MInstSSR : public MemoryInst {
public:
    int qIdx;
    MInstSSR(int qIdx) : MemoryInst(MOp_SSR), qIdx(qIdx) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "SSR " << qIdx << std::string(10, ' ');
    }
};

// Shuffle Swap Col (SSC)
class MInstSSC : public MemoryInst {
public:
    int qIdx;
    MInstSSC(int qIdx) : MemoryInst(MOp_SSC), qIdx(qIdx) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "SSC " << qIdx << std::string(10, ' ');
    }
};

// Full Swap Row (FSR)
class MInstFSR : public MemoryInst {
public:
    int qIdx;
    int cycle;

    MInstFSR(int qIdx, int cycle)
        : MemoryInst(MOp_FSR), qIdx(qIdx), cycle(cycle) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "FSR <cycle=" << cycle << "> " << qIdx;
    }
};

// Full Swap Col (FSC)
class MInstFSC : public MemoryInst {
public:
    int qIdx;
    int cycle;

    MInstFSC(int qIdx, int cycle)
        : MemoryInst(MOp_FSC), qIdx(qIdx), cycle(cycle) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "FSC <cycle=" << cycle << "> " << qIdx;
    }
};

// External (EXT)
class MInstEXT : public MemoryInst {
public:
    std::vector<int> flags;

    MInstEXT(std::initializer_list<int> flags)
        : MemoryInst(MOp_EXT), flags(flags) {}
    MInstEXT(const std::vector<int>& flags)
        : MemoryInst(MOp_EXT), flags(flags) {}

    std::ostream& print(std::ostream& os) const override;
};

class GateInst {
private:
    GInstKind gKind;
public:
    GateBlock* block;

    GateInst(GInstKind gKind, GateBlock* block = nullptr)
        : gKind(gKind), block(block) {}
    
    virtual ~GateInst() = default;

    GInstKind getKind() const { return gKind; }

    bool isNull() const { return getKind() == GOp_NUL; }
    virtual std::ostream& print(std::ostream& os) const {
        assert(false && "Calling from base class");
        return os;
    }
};

class GInstNUL : public GateInst {
public:
    GInstNUL() : GateInst(GOp_NUL, nullptr) {}

    std::ostream& print(std::ostream& os) const override {
        return os << "NUL";
    }
};

// Single Qubit Gate (SQ)
class GInstSQ : public GateInst {
public:
    GInstSQ(GateBlock* block) : GateInst(GOp_SQ, block) {}
    
    std::ostream& print(std::ostream& os) const override;
};

// Unitary Permutation Gate (UP)
class GInstUP : public GateInst {
public:
    GInstUP(GateBlock* block) : GateInst(GOp_UP, block) {}
    
    std::ostream& print(std::ostream& os) const override;
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

    FPGACostConfig(double tExtMemOp, double tMemOpOnly, double tRealGate,
                   double tUnitaryPerm, double tGeneral)
        : tExtMemOp(tExtMemOp),
          tMemOpOnly(tMemOpOnly),
          tRealGate(tRealGate),
          tUnitaryPerm(tUnitaryPerm),
          tGeneral(tGeneral) {}
};


class Instruction {
public:
    std::unique_ptr<MemoryInst> mInst;
    std::unique_ptr<GateInst> gInst;

    Instruction(std::unique_ptr<MemoryInst> _mInst,
                std::unique_ptr<GateInst> _gInst) {
        setMInst(std::move(_mInst));
        setGInst(std::move(_gInst));
    }

    std::ostream& print(std::ostream& os) const {
        mInst->print(os) << " : ";
        gInst->print(os) << "\n";
        return os;
    }

    void setMInst(std::unique_ptr<MemoryInst> inst) {
        if (inst) {
            mInst = std::move(inst);
            return;
        }
        mInst = std::make_unique<MInstNUL>();
    }

    void setGInst(std::unique_ptr<GateInst> inst) {
        if (inst) {
            gInst = std::move(inst);
            return;
        }
        gInst = std::make_unique<GInstNUL>();
    }

    double cost(const FPGACostConfig&) const;
};

struct FPGAInstGenConfig {
public:
    int nLocalQubits;
    int gridSize;

    // If off, apply sequential instruction generation on the default order of
    // blocks present in CircuitGraph
    bool selectiveGenerationMode;

    int getNOnChipQubits() const {
        return nLocalQubits + 2 * gridSize;
    }

    FPGAInstGenConfig(int nLocalQubits, int gridSize,
                      bool selectiveGenerationMode = true) 
        : nLocalQubits(nLocalQubits),
          gridSize(gridSize),
          selectiveGenerationMode(selectiveGenerationMode) {}
};

// top-level function to generate FPGA instructions from a CircuitGraph
std::vector<Instruction> genInstruction(
        const CircuitGraph&, const FPGAInstGenConfig&);

}; // namespace saot::fpga

#endif // SAOT_FPGAINST_H