#include "saot/FPGAInst.h"
#include "saot/CircuitGraph.h"

using namespace saot;
using namespace saot::fpga;

std::ostream& MInstEXT::print(std::ostream& os) const {
    os << "EXT ";
    utils::printVector(flags, os);
    return os;
}

std::ostream& GInstSQ::print(std::ostream& os) const {
    os << "UP<id=" << block->id << "> ";
    for (const auto& data : block->dataVector)
        os << data.qubit << " ";
    return os;
}

std::ostream& GInstUP::print(std::ostream& os) const {
    os << "UP<id=" << block->id << "> ";
    for (const auto& data : block->dataVector)
        os << data.qubit << " ";
    return os;
}

double Instruction::cost(const FPGACostConfig& config) const {
    if (mInst->getKind() == MOp_EXT) {
        if (dynamic_cast<MInstEXT&>(*mInst).flags[0] < 7)
            return 2 * config.tExtMemOp;
        return config.tExtMemOp;
    }

    if (gInst->isNull()) {
        assert(!mInst->isNull());
        return config.tMemOpOnly;
    }
    
    if (gInst->getKind() == GOp_UP)
        return config.tUnitaryPerm;
    assert(gInst->getKind() == GOp_SQ);

    if (fpga::getFPGAGateCategory(*gInst->block->quantumGate) & fpga::fpgaRealOnly)
        return config.tRealGate;
    return config.tGeneral;
}

// helper methods to saot::fpga::genInstruction
namespace {

enum QubitKind : int {
    QK_Unknown = -1,

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

    QubitStatus() : kind(QK_Unknown), kindIdx(0) {}
    QubitStatus(QubitKind kind, int kindIdx) : kind(kind), kindIdx(kindIdx) {}

    std::ostream& print(std::ostream& os) const {
        os << "(";
        switch (kind) {
            case QK_Local: os << "loc"; break;
            case QK_Row: os << "row"; break;
            case QK_Col: os << "col"; break;
            case QK_Depth: os << "dep"; break;
            case QK_OffChip: os << "ext"; break;
            default: break;
        }
        os << ", " << kindIdx << ")";
        return os;
    }
};

// 0, 1, 2, 4
int getNumberOfFullSwapCycles(int kindIdx) {
    return (1 << kindIdx) >> 1;
}

class InstGenState {
private:
    enum on_chip_flag_t {
        OCF_NotInited,  // not initialized
        OCF_OnChip,     // on-chip
        OCF_OffChip,    // off-chip
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
        on_chip_flag_t isOnChip;
        available_block_kind_t kind;

        available_blocks_t(GateBlock* block,
                           on_chip_flag_t isOnChip = OCF_NotInited,
                           available_block_kind_t kind = ABK_NotInited)
            : block(block), isOnChip(isOnChip), kind(kind) {}
    };

    void init(const CircuitGraph& graph) {
        // initialize qubit statuses
        int nOnChipQubits = config.getNOnChipQubits();
        for (int i = 0; i < config.nLocalQubits; i++) // local
            qubitStatuses[i] = QubitStatus(QK_Local, i);
        for (int i = 0; i < config.gridSize; i++) // row
            qubitStatuses[config.nLocalQubits + i] = QubitStatus(QK_Row, i);
        for (int i = 0; i < config.gridSize; i++) // col
            qubitStatuses[config.nLocalQubits + config.gridSize + i] = QubitStatus(QK_Col, i);
        for (int i = 0; i < nqubits - nOnChipQubits; i++) // off-chip
            qubitStatuses[nOnChipQubits + i] = QubitStatus(QK_OffChip, i);

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
    const FPGAInstGenConfig& config;
    int nrows;
    int nqubits;
    std::vector<QubitStatus> qubitStatuses;
    std::vector<GateBlock*> tileBlocks;
    // unlockedRowIndices[q] gives the index of the last unlocked row in wire q
    std::vector<int> unlockedRowIndices;
    std::vector<available_blocks_t> availables;

    InstGenState(const CircuitGraph& graph, const FPGAInstGenConfig& config)
            : config(config),
              nrows(graph.tile().size()),
              nqubits(graph.nqubits),
              qubitStatuses(graph.nqubits),
              tileBlocks(graph.tile().size() * nqubits),
              unlockedRowIndices(nqubits),
              availables() { init(graph); }

    std::ostream& printQubitStatuses(std::ostream& os) const {
        auto it = qubitStatuses.cbegin();
        it->print(os << "0:");
        int i = 1;
        while (++it != qubitStatuses.cend())
            it->print(os << ", " << i << ":");
        return os;
    }

    // Update availables depending on qubitKinds. This function should be called
    // whenever qubitStatuses is changed
    void updateAvailables() {
        const auto updateOnChipFlag = [&](available_blocks_t& available) {
            for (const auto& data : available.block->dataVector) {
                if (qubitStatuses[data.qubit].kind == QK_OffChip) {
                    available.isOnChip = OCF_OffChip;
                    return;
                }
            }
            available.isOnChip = OCF_OnChip;
        };

        const auto updateKind = [&](available_blocks_t& available) {
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
        };

        for (auto& available : availables) {
            assert(available.block);
            updateOnChipFlag(available);
            updateKind(available);
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

    GateBlock* findOnChipBlockWithKind(available_block_kind_t kind) const {
        for (const auto& candidate : availables) {
            if (candidate.isOnChip != OCF_OnChip)
                continue;
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

        // This method will update vacantMemIdx = idx + 1
        const auto writeMemInst = [&](int idx, std::unique_ptr<MemoryInst> inst) {
            if (idx < instructions.size()) {
                assert(instructions[idx].mInst->isNull());
                instructions[idx].setMInst(std::move(inst));
            } else {
                assert(idx == instructions.size());
                instructions.emplace_back(std::move(inst), nullptr);
            }
            vacantMemIdx = idx + 1;
        };

        const auto generateFullSwap = [&](int localQ, int nonLocalQ) {
            assert(qubitStatuses[localQ].kind == QK_Local);
            assert(qubitStatuses[nonLocalQ].kind != QK_Local);
            const int fullSwapQIdx = qubitStatuses[nonLocalQ].kindIdx;
            const int nFSCycles = getNumberOfFullSwapCycles(fullSwapQIdx);
            const int shuffleSwapQIdx = qubitStatuses[localQ].kindIdx;

            int insertIdx = std::max(vacantMemIdx, sqGateBarrierIdx);
            // full swaps
            for (int cycle = 0; cycle < nFSCycles; cycle++) {
                if (qubitStatuses[nonLocalQ].kind == QK_Row)
                    writeMemInst(insertIdx++, std::make_unique<MInstFSR>(fullSwapQIdx, cycle));
                else
                    writeMemInst(insertIdx++, std::make_unique<MInstFSC>(fullSwapQIdx, cycle));
            }
            // shuffle swap
            if (qubitStatuses[nonLocalQ].kind == QK_Row)
                writeMemInst(insertIdx++, std::make_unique<MInstSSR>(shuffleSwapQIdx));
            else
                writeMemInst(insertIdx++, std::make_unique<MInstSSC>(shuffleSwapQIdx));

            // swap qubit statuses
            if (fullSwapQIdx != 0) {
                // permute nonLocalQ -> kind[0] -> localQ
                auto it = std::find_if(qubitStatuses.begin(), qubitStatuses.end(),
                    [kind=qubitStatuses[nonLocalQ].kind](const QubitStatus& S) {
                        return S.kind == kind && S.kindIdx == 0;
                    });
                assert(it != qubitStatuses.end());
                auto tmp = *it;
                *it = qubitStatuses[nonLocalQ];
                qubitStatuses[nonLocalQ] = qubitStatuses[localQ];
                qubitStatuses[localQ] = tmp;
            }
            else {
                // swap nonLocalQ and localQ
                auto tmp = qubitStatuses[localQ];
                qubitStatuses[localQ] = qubitStatuses[nonLocalQ];
                qubitStatuses[nonLocalQ] = tmp;
            }
            updateAvailables();
        };

        const auto generateUPBlock = [&](GateBlock* b) {
            popBlock(b);

            if (vacantGateIdx == instructions.size()) {
                instructions.emplace_back(nullptr, std::make_unique<GInstUP>(b));
            } else {
                auto& inst = instructions[vacantGateIdx];
                assert(inst.gInst->isNull());
                inst.setGInst(std::make_unique<GInstUP>(b));
            }
            ++vacantGateIdx;
        };

        const auto generateLocalSQBlock = [&](GateBlock* b) {
            popBlock(b);
            assert(b->quantumGate->qubits.size() == 1 &&
                "SQ Block has more than 1 target qubits?");
            auto qubit = b->quantumGate->qubits[0];
            assert(qubitStatuses[qubit].kind == QK_Local);

            instructions.emplace_back(nullptr, std::make_unique<GInstSQ>(b));
            vacantGateIdx = instructions.size();
            sqGateBarrierIdx = vacantGateIdx - 1;
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

        const auto generateOnChipReassignment = [&]() {
            std::vector<int> priorities;
            priorities.reserve(nqubits);

            auto availablesCopy(availables);
            // prioritize assigning SQ gates as local
            while (!availablesCopy.empty()) {
                auto it = std::find_if(availablesCopy.begin(), availablesCopy.end(),
                [](const available_blocks_t& avail) {
                    return avail.kind == ABK_LocalSQ || avail.kind == ABK_NonLocalSQ;
                });
                if (it == availablesCopy.end())
                    break;
                assert(it->block->nqubits() == 1);
                int q = it->block->dataVector[0].qubit;
                utils::pushBackIfNotInVector(priorities, q);
                availablesCopy.erase(it);
            }
            // no SQ gates, prioritize UP gates
            for (const auto& avail : availablesCopy) {
                for (const auto& data : avail.block->dataVector)
                    utils::pushBackIfNotInVector(priorities, data.qubit);
            }
            // fill up priorities vector
            for (int q = 0; q < nqubits; q++)
                utils::pushBackIfNotInVector(priorities, q);

            // update qubitStatuses
            assert(utils::isPermutation(priorities));
            int nOnChipQubits = config.getNOnChipQubits();
            for (int i = 0; i < config.nLocalQubits; i++) // local
                qubitStatuses[priorities[i]] = QubitStatus(QK_Local, i);
            for (int i = 0; i < config.gridSize; i++) // row
                qubitStatuses[priorities[config.nLocalQubits + i]] = QubitStatus(QK_Row, i);
            for (int i = 0; i < config.gridSize; i++) // col
                qubitStatuses[priorities[config.nLocalQubits + config.gridSize + i]] = QubitStatus(QK_Col, i);
            for (int i = 0; i < nqubits - nOnChipQubits; i++) // off-chip
                qubitStatuses[priorities[nOnChipQubits + i]] = QubitStatus(QK_OffChip, i);

            writeMemInst(std::max(vacantMemIdx, sqGateBarrierIdx), std::make_unique<MInstEXT>(priorities));
            updateAvailables();
        };

        while (!availables.empty()) {
            // TODO: handle non-comp gates (omit them for now)
            bool nonCompFlag = false;
            for (const auto& avail : availables) {
                // omit non-comp gates
                if (fpga::getFPGAGateCategory(*avail.block->quantumGate) & fpga::fpgaNonComp) {
                    // std::cerr << "Ignored block " << avail.block->id << " because it is non-comp\n";
                    popBlock(avail.block);
                    nonCompFlag = true;
                    break;
                }
            }
            if (nonCompFlag)
                continue;
            
            if (!config.selectiveGenerationMode) {
                assert(false && "Not Implemented");
                continue;
            }
            
            // TODO: optimize this traversal
            if (auto* b = findOnChipBlockWithKind(ABK_LocalSQ))
                generateLocalSQBlock(b);
            else if (auto* b = findOnChipBlockWithKind(ABK_UnitaryPerm))
                generateUPBlock(b);
            else if (auto* b = findOnChipBlockWithKind(ABK_NonLocalSQ))
                generateNonLocalSQBlock(b);
            else // no onChipBlock
                generateOnChipReassignment();
        }
        return instructions;
    }
};

} // anonymous namespace

std::vector<Instruction> saot::fpga::genInstruction(
        const CircuitGraph& graph, const FPGAInstGenConfig& config) {
    InstGenState state(graph, config);

    return state.generate();
}

