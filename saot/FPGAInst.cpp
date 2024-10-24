#include "saot/FPGAInst.h"
#include "saot/CircuitGraph.h"

using namespace saot;
using namespace saot::fpga;

const FPGAInstGenConfig FPGAInstGenConfig::Grid2x2 = {
    .gridSize = 2
};

const FPGAInstGenConfig FPGAInstGenConfig::Grid3x3 = {
    .gridSize = 3
};

const FPGAInstGenConfig FPGAInstGenConfig::Grid4x4 = {
    .gridSize = 4
};


std::ostream& GateInst::print(std::ostream& os) const {
    const auto printQubits = [&]() {
        if (qubits.empty())
            return;
        auto it = qubits.begin();
        os << *it;
        while (++it != qubits.end())
            os << " " << *it;
    };

    switch (op) {
    case GOp_NUL:
        return os << "NUL";
    case GOp_SQ: {
        os << "GSQ <id=" << gateID << "> ";
        printQubits();
        return os;
    }
    case GOp_UP: {
        os << "GUP <id=" << gateID << ", size=" << qubits.size() << "> ";
        printQubits();
        return os;
    }
    default:
        return os << "<Unknown GateOp>";
    }
    return os;
}

std::ostream& MemoryInst::print(std::ostream& os) const {
    switch (op) {
    case MOp_NUL:
        return os << "NUL" << std::string(12, ' ');
    case MOp_SSR:
        return os << "SSR " << qIdx << std::string(10, ' ');
    case MOp_SSC:
        return os << "SSC " << qIdx << std::string(10, ' ');
    case MOp_FSC:
        return os << "FSC <cycle=" << cycle << "> " << qIdx;
    case MOp_FSR:
        return os << "FSR <cycle=" << cycle << "> " << qIdx;
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
    QK_Depth = 3,
    QK_OffChip = 4,
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
        // initialize qubit statuses
        qubitStatuses.reserve(nqubits);
        int nLocalQubits = nqubits - 2 * gridSize;
        assert(nLocalQubits > 0);
        for (int i = 0; i < nLocalQubits; i++)
            qubitStatuses[i] = QubitStatus(QK_Local, i);
        for (int i = 0; i < gridSize; i++)
            qubitStatuses[nLocalQubits + i] = QubitStatus(QK_Row, i);
        for (int i = 0; i < gridSize; i++)
            qubitStatuses[nLocalQubits + gridSize + i] = QubitStatus(QK_Col, 1);   
        // initialize node state
        int row = 0;
        for (auto it = graph.tile().begin(); it != graph.tile().end(); it++, row++) {
            for (unsigned q = 0; q < nqubits; q++)
                tileBlocks[nqubits * row + q] = (*it)[q];
        }
        // initialize unlockedRowIndices
        for (unsigned q = 0; q < nqubits; q++) {
            for (row = 0; row < nrows; row++) {
                if (tileBlocks[nqubits * row + q] != nullptr)
                    break;
            }
            unlockedRowIndices[q] = row;
        }
        // initialize availables
        for (unsigned q = 0; q < nqubits; q++) {
            row = unlockedRowIndices[q];
            if (row >= nrows)
                continue;
            auto* cddBlock = tileBlocks[nqubits * row + q];
            assert(cddBlock);
            if (std::find_if(availables.begin(), availables.end(),
                    [&cddBlock](const available_blocks_t& avail) {
                        return avail.block == cddBlock;
                    }) != availables.end()) {
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
        updateAvailables();
    }
public:
    int gridSize;
    int nrows;
    int nqubits;
    std::vector<QubitStatus> qubitStatuses;
    std::vector<GateBlock*> tileBlocks;
    // unlockedRowIndices[q] gives the index of the last unlocked row in wire q
    std::vector<int> unlockedRowIndices;
    std::vector<available_blocks_t> availables;

    InstGenState(const CircuitGraph& graph, int gridSize)
            : gridSize(gridSize),
              nrows(graph.tile().size()),
              nqubits(graph.nqubits),
              qubitStatuses(),
              tileBlocks(graph.tile().size() * nqubits),
              unlockedRowIndices(nqubits),
              availables() { init(graph); }

    // update availables depending on qubitKinds
    void updateAvailables() {
        for (auto& available : availables) {
            if (available.kind == ABK_NotInited) {
                if (available.block->quantumGate->isConvertibleToUnitaryPermGate())
                    available.kind = ABK_UnitaryPerm;
                else {
                    const auto& qubit = available.block->quantumGate->qubits[0];
                    assert(available.block->quantumGate->qubits.size() == 1);
                    available.kind = (qubitStatuses[qubit].kind == QK_Local) ? ABK_LocalSQ : ABK_NonLocalSQ;
                }
            }
            // only need to update single-qubit blocks now
            else if (available.kind == ABK_LocalSQ || available.kind == ABK_NonLocalSQ) {
                const auto& qubit = available.block->quantumGate->qubits[0];
                assert(available.block->quantumGate->qubits.size() == 1);
                available.kind = (qubitStatuses[qubit].kind == QK_Local) ? ABK_LocalSQ : ABK_NonLocalSQ;
            }
        }
    }

    void popBlock(GateBlock* block) {
        auto it = std::find_if(availables.begin(), availables.end(),
            [&block](const available_blocks_t& avail) {
                return avail.block == block;
            });
        assert(it != availables.end());
        availables.erase(it);

        // grab next availables
        std::vector<GateBlock*> candidateBlocks;
        for (const auto& data : block->dataVector) {
            const auto& qubit = data.qubit;

            GateBlock* cddBlock = nullptr;
            for (auto& updatedRow = ++unlockedRowIndices[qubit]; updatedRow < nrows; ++updatedRow) {
                auto idx = nqubits * updatedRow + qubit;
                cddBlock = tileBlocks[idx];
                if (cddBlock)
                    break;
            }
            if (cddBlock &&
                    std::find(candidateBlocks.begin(), candidateBlocks.end(), cddBlock) == candidateBlocks.end())
                candidateBlocks.push_back(cddBlock);
        }
        for (const auto& b : candidateBlocks) {
            bool insertFlag = true;
            auto row = unlockedRowIndices[b->dataVector[0].qubit];
            for (const auto& data : b->dataVector) {
                if (unlockedRowIndices[data.qubit] != row) {
                    insertFlag = false;
                    break;
                }
            }
            if (insertFlag)
                availables.emplace_back(b);
        }
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
        // The minimum indices at which we can insert mem / gate instructions
        int vacantMemIdx = 0;
        int vacantGateIdx = 0;
        int sqGateBarrierIdx = 0; // single-qubit gate

        const auto writeMemInst = [&](int idx, const MemoryInst& inst) {
            if (idx < instructions.size()) {
                assert(instructions[idx].memInst.isNull());
                instructions[idx].memInst = inst;
            } else {
                assert(idx == instructions.size());
                instructions.emplace_back(inst, GateInst());
            }
        };

        const auto generateFullSwap = [&](int localQ, int nonLocalQ) {
            assert(qubitStatuses[localQ].kind == QK_Local);
            assert(qubitStatuses[nonLocalQ].kind != QK_Local);
            const int fullSwapQIdx = qubitStatuses[nonLocalQ].kindIdx;
            const int nFSCycles = getNumberOfFullSwapCycles(fullSwapQIdx);
            const int shuffleSwapQIdx = qubitStatuses[localQ].kindIdx;
            const MemoryOp fullSwapOp = 
                (qubitStatuses[nonLocalQ].kind == QK_Row) ? MOp_FSR : MOp_FSC;
            const MemoryOp shuffleSwapOp = 
                (qubitStatuses[nonLocalQ].kind == QK_Row) ? MOp_SSR : MOp_SSC;
            int insertIdx = std::max(vacantMemIdx, sqGateBarrierIdx);
            // full swaps + shuffle swap
            for (int cycle = 0; cycle < nFSCycles; cycle++)
                writeMemInst(insertIdx++, MemoryInst(fullSwapOp, fullSwapQIdx, cycle));
            writeMemInst(insertIdx++, MemoryInst(shuffleSwapOp, shuffleSwapQIdx));

            vacantMemIdx = insertIdx;
            // swap qubit statuses
            auto tmp = qubitStatuses[localQ];
            qubitStatuses[localQ] = qubitStatuses[nonLocalQ];
            qubitStatuses[nonLocalQ] = tmp;
            updateAvailables();
        };

        const auto generateUPBlock = [&](GateBlock* b) {
            popBlock(b);

            if (vacantGateIdx == instructions.size()) {
                instructions.emplace_back(
                    MemoryInst(MOp_NUL), GateInst(GOp_UP, b->id, b->quantumGate->qubits));
            } else {
                auto& inst = instructions[vacantGateIdx];
                assert(inst.gateInst.isNull());
                inst.gateInst = GateInst(GOp_UP, b->id, b->quantumGate->qubits);
            }
            ++vacantGateIdx;
        };

        const auto generateLocalSQBlock = [&](GateBlock* b) {
            popBlock(b);
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitStatuses[qubit].kind == QK_Local);

            instructions.emplace_back(MemoryInst(), GateInst(GOp_SQ, b->id, {qubit}));
            sqGateBarrierIdx = instructions.size();
            vacantGateIdx = sqGateBarrierIdx;
        };

        const auto generateNonLocalSQBlock = [&](GateBlock* b) {
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitStatuses[qubit].kind != QK_Local);
            // TODO: the ideal case is after full swap, there is a local SQ
            // block. However, we need deeper search since potentially many
            // UP gates are to be applied together with full swap insts.

            // For now, we always use the first (least significant) local qubit.
            for (int localQ = 0; localQ < nqubits; localQ++) {
                if (qubitStatuses[localQ].kind == QK_Local) {
                    generateFullSwap(localQ, qubit);
                    break;
                }
            }
            generateLocalSQBlock(b);
        };

        while (!availables.empty()) {
            // if (vacantMemIdx < vacantGateIdx) {
            //     if (auto* b = findBlockWithKind(ABK_LocalSQ))
            //         generateLocalSQBlock(b);
            //     else if (auto* b = findBlockWithKind(ABK_UnitaryPerm))
            //         generateUPBlock(b);
            //     else if (auto* b = findBlockWithKind(ABK_NonLocalSQ))
            //         generateNonLocalSQBlock(b);
            //     else
            //         assert(false && "Unreachable");
            // }
            // else {
                if (auto* b = findBlockWithKind(ABK_LocalSQ))
                    generateLocalSQBlock(b);
                else if (auto* b = findBlockWithKind(ABK_UnitaryPerm))
                    generateUPBlock(b);
                else if (auto* b = findBlockWithKind(ABK_NonLocalSQ))
                    generateNonLocalSQBlock(b);
                else
                    assert(false && "Unreachable");
            // }
        }
        return instructions;
    }
};

} // anonymous namespace

std::vector<Instruction> saot::fpga::genInstruction(
        const CircuitGraph& graph, const FPGAInstGenConfig& config) {
    InstGenState state(graph, config.gridSize);

    return state.generate();
}
