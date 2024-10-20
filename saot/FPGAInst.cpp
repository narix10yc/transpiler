#include "saot/FPGAInst.h"
#include "saot/CircuitGraph.h"

using namespace saot;
using namespace saot::fpga;

const FPGAInstGenConfig FPGAInstGenConfig::Default = {
    .gridSize = 4
};


std::ostream& GateInst::print(std::ostream& os) {
    switch (op) {
    case GOp_NUL:
        return os << "NUL";
    case GOp_SQ:
        return os << "GSQ " << arg;
    case GOp_UP:
        return os << "GUP " << arg;
    default:
        return os << "<Unknown GateOp>";
    }
    return os;
}

std::ostream& MemoryInst::print(std::ostream& os) {
    switch (op) {
    case MOp_NUL:
        return os << "NUL";
    case MOp_SSR:
        return os << "SSR " << qIdx << " " << cycle;
    case MOp_SSC:
        return os << "SSC " << qIdx << " " << cycle;
    case MOp_FSC:
        return os << "FSC " << qIdx << " " << cycle;
    case MOp_FSR:
        return os << "FSR " << qIdx << " " << cycle;
    default:
        return os << "<Unknown MemOp>";
    }
}


// helper methods to saot::fpga::genInstruction
namespace {

enum QubitKind : int {
    QK_Local = 0,
    QK_Row = 1,
    QK_Col = 2,
};

struct QubitStatus {
    QubitKind kind;
    // the index of this qubit among all qubits with the same kind
    int kindIdx;

    QubitStatus(QubitKind kind, int kindIdx) : kind(kind), kindIdx(kindIdx) {}
};

// 0, 1, 2, 4
int getNumberOfFullSwapCycles(int kindIdx) {
    return (1 << kindIdx) >> 1;
}

class InstGenState {
private:
    struct pending_instruction_t {
        MemoryInst memInst;
        GateInst gateInst;
        GateBlock* block;

        pending_instruction_t(const MemoryInst& m, const GateInst& g, GateBlock* b)
            : memInst(m), gateInst(g), block(b) {}
    };

    enum available_block_kind_t {
        ABK_LocalSQ,     // local single-qubit
        ABK_NonLocalSQ,  // non-local single-qubit
        ABK_UnitaryPerm, // unitary permutation
        ABK_NonComp,     // non-computational
        ABK_NotInited,   // not initialized
    };

    struct available_blocks_t {
        GateBlock* block;
        available_block_kind_t kind;

        available_blocks_t(GateBlock* block, available_block_kind_t kind = ABK_NotInited)
            : block(block), kind(kind) {}
    };

    void init(const CircuitGraph& graph) {
        // initialize qubit kind
        int nLocalQubits = nqubits - 2 * gridSize;
        assert(nLocalQubits > 0);
        for (int i = 0; i < nLocalQubits; i++)
            qubitKinds[i] = QK_Local;
        for (int i = 0; i < gridSize; i++)
            qubitKinds[nLocalQubits + i] = QK_Row;
        for (int i = 0; i < gridSize; i++)
            qubitKinds[nLocalQubits + gridSize + i] = QK_Col;   
        // initialize node state
        int row = 0;
        for (auto it = graph.tile().begin(); it != graph.tile().end(); it++, row++) {
            for (unsigned q = 0; q < nqubits; q++)
                tileBlocks[nqubits * row + q] = (*it)[q];
        }
        // initialize unlockedRowIndices
        for (unsigned q = 0; q < nqubits; q++) {
            for (unsigned row = 0; row < nrows; row++) {
                if (tileBlocks[nqubits * row + q] != nullptr)
                    break;
            }
            unlockedRowIndices[q] = row;
        }
        // initialize availables
        for (unsigned q = 0; q < nqubits; q++) {
            row = unlockedRowIndices[q] + 1;
            if (row >= nrows)
                continue;
            unlockedRowIndices[q]++;
            auto* cddBlock = tileBlocks[nqubits * row + q];
            if (cddBlock == nullptr)
                continue;
            if (std::find(availables.begin(), availables.end(),
                    [&cddBlock](const available_blocks_t avail) {
                        return avail.block == cddBlock;
                    }) == availables.end()) {
                continue;
            }
            
            bool acceptFlag = true;
            for (const auto& bData : cddBlock->dataVector) {
                if (unlockedRowIndices[bData.qubit] < row) {
                    acceptFlag = false;
                    break;
                }
            }
            if (acceptFlag)
                availables.emplace_back(cddBlock);
        }
    }
public:
    int gridSize;
    int nrows;
    int nqubits;
    std::vector<QubitKind> qubitKinds;
    std::vector<GateBlock*> tileBlocks;
    // unlockedRowIndices[q] gives the index of the last unlocked row in wire q
    std::vector<int> unlockedRowIndices;
    std::vector<available_blocks_t> availables;

    InstGenState(const CircuitGraph& graph, int gridSize)
            : gridSize(gridSize),
              nrows(graph.tile().size()),
              nqubits(graph.nqubits), 
              tileBlocks(graph.tile().size() * graph.nqubits),
              unlockedRowIndices(graph.tile().size()),
              availables() { init(graph); }

    QubitStatus getQubitState(int q) const {
        assert(q < qubitKinds.size());
        auto kind = qubitKinds[q];
        int count = 0;
        for (int i = 0; i < q; i++) {
            if (qubitKinds[i] == kind)
                count++;
        }
        return QubitStatus(kind, count);
    }

    // update availables depending on qubitKinds
    void updateAvailables() {
        for (auto& available : availables) {
            if (available.kind == ABK_NotInited) {
                if (available.block->quantumGate->isConvertibleToUnitaryPermGate())
                    available.kind = ABK_UnitaryPerm;
                else {
                    const auto& qubit = available.block->quantumGate->qubits[0];
                    assert(available.block->quantumGate->qubits.size() == 1);
                    available.kind = (qubitKinds[qubit] == QK_Local) ? ABK_LocalSQ : ABK_NonLocalSQ;
                }
            }
            // only need to update single-qubit blocks now
            else if (available.kind == ABK_LocalSQ || available.kind == ABK_NonLocalSQ) {
                const auto& qubit = available.block->quantumGate->qubits[0];
                assert(available.block->quantumGate->qubits.size() == 1);
                available.kind = (qubitKinds[qubit] == QK_Local) ? ABK_LocalSQ : ABK_NonLocalSQ;
            }
        }
    }

    void popBlock(GateBlock* block) {
        auto it = std::find(availables.begin(), availables.end(), block);
        assert(it != availables.end());

        std::vector<GateBlock*> candidateBlocks;
        int row = unlockedRowIndices[block->dataVector[0].qubit];
        for (const auto& data : block->dataVector) {
            const auto& qubit = data.qubit;
            assert(row == unlockedRowIndices[qubit]);

            for (auto& updatedRow = unlockedRowIndices[qubit]; updatedRow < nrows; ++updatedRow) {
                auto* cddBlock = tileBlocks[nrows * updatedRow + qubit];
                if (cddBlock && cddBlock != block && 
                        std::find(candidateBlocks.begin(), candidateBlocks.end(), cddBlock) == candidateBlocks.end()) {
                    candidateBlocks.push_back(cddBlock);
                    break;
                }
            }
        }
        availables.erase(it);
        for (auto& b : candidateBlocks)
            availables.emplace_back(b);
        updateAvailables();
    }

    GateBlock* findBlockWithKind(available_block_kind_t kind) const {
        for (const auto& candidate : availables) {
            if (candidate.kind == kind)
                return candidate.block;
        }
        return nullptr;
    }

    std::vector<Instruction> generate() {
        std::vector<Instruction> instructions;
        std::deque<pending_instruction_t> pendingInsts;

        const auto popPendingInstruction = [&]() {
            assert(!pendingInsts.empty());

        };

        const auto generateFullSwap = [&](int localQ, int nonLocalQ) {
            assert(qubitKinds[localQ] == QK_Local);
            assert(qubitKinds[nonLocalQ] != QK_Local);
            const int nFSCycles = getNumberOfFullSwapCycles(getQubitState(nonLocalQ).kindIdx);
            const int fullSwapQIdx = getQubitState(localQ).kindIdx;
            const MemoryOp fullSwapOp = 
                (qubitKinds[nonLocalQ] == QK_Row) ? MOp_FSR : MOp_FSC;
            for (int cycle = 0; cycle < nFSCycles; cycle++) {
                pendingInsts.emplace_back(
                    MemoryInst(fullSwapOp, fullSwapQIdx, cycle), 
                    GateInst(GOp_NUL), nullptr);
            }
            qubitKinds[localQ] = qubitKinds[nonLocalQ];
            qubitKinds[nonLocalQ] = QK_Local;
            updateAvailables();
        };

        const auto generateUPBlock = [&](GateBlock* b) {

        };

        const auto generateLocalSQBlock = [&](GateBlock* b) {
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitKinds[qubit] == QK_Local);
            // auto qubitState = getQubitState(qubit);
            while (true) {
                if (pendingInsts.empty()) {
                    pendingInsts.emplace_back(
                    MemoryInst(MOp_NUL), GateInst(GOp_UP, qubit), b);
                    return;
                }
                auto& frontPendingInst = pendingInsts.front();
                if (frontPendingInst.gateInst.op == GOp_NUL) {
                    frontPendingInst.gateInst = GateInst(GOp_UP, qubit);
                    frontPendingInst.block = b;
                    popPendingInstruction();
                    return;
                }
                popPendingInstruction();
            }
        };

        const auto generateNonLocalSQBlock = [&](GateBlock* b) {
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitKinds[qubit] != QK_Local);
            // TODO: the ideal case is after full swap, there is a local SQ
            // block. However, we need deeper search since potentially many
            // UP gates are to be applied together with full swap insts.

            // For now, we always use the first (least significant) local qubit.
            for (int localQ = 0; localQ < nqubits; localQ++) {
                if (qubitKinds[localQ] == QK_Local) {
                    generateFullSwap(localQ, qubit);
                    break;
                }
            }
            pendingInsts.emplace_back(
                MemoryInst(MOp_NUL), GateInst(GOp_UP, qubit), b);
        };

        while (!availables.empty()) {
            if (pendingInsts.empty()) {
                // prioritize non-local SQ than local SQ than UP
                if (auto* b = findBlockWithKind(ABK_NonLocalSQ))
                    generateNonLocalSQBlock(b);
                else if (auto* b = findBlockWithKind(ABK_LocalSQ))
                    generateLocalSQBlock(b);
                else if (auto* b = findBlockWithKind(ABK_UnitaryPerm))
                    generateUPBlock(b);
                else
                    assert(false && "Unreachable");

                continue;
            }
            // if pendingInsts is non-empty
            assert(pendingInsts.front().memInst.op != MOp_NUL || 
                   pendingInsts.front().gateInst.op != GOp_NUL);
            if (pendingInsts.front().memInst.op == MOp_NUL) {

                continue;
            }


        }
    }
    
};


} // anonymous namespace

std::vector<Instruction> genInstruction(
        const CircuitGraph& graph, const FPGAInstGenConfig& config) {
    InstGenState state(graph, config.gridSize);

    return state.generate();
}
