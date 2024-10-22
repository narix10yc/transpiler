#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/QuantumGate.h"
#include "utils/iocolor.h"

using namespace saot;
using namespace IOColor;

FPGAFusionConfig FPGAFusionConfig::Default = FPGAFusionConfig {
            .maxUnitaryPermutationSize = 5,
            .ignoreSingleQubitNonCompGates = true,
            .multiTraverse = true,
        };

FPGAGateCategory getFPGAGateCategory(const QuantumGate& gate) {
    switch (gate.gateMatrix.gateTy){
        case gX: return fpgaSingleQubitNonComp;
        case gY: return fpgaSingleQubitNonComp;
        case gZ: return fpgaSingleQubitNonComp;
        case gP: return fpgaSingleQubitUnitaryPerm;
        case gH: return fpgaSingleQubit;
        case gCX: return fpgaNonComp;
        case gCZ: return fpgaNonComp;
        case gCP: return fpgaUnitaryPerm;
        default: break; 
    }

    if (const auto* p = std::get_if<GateMatrix::up_matrix_t>(&gate.gateMatrix._matrix))
        return fpgaUnitaryPerm;

    // TODO: handle general gates
    return fpgaGeneral;
}

inline bool isUnitaryPermBlock(const GateBlock* block) {
    assert(block != nullptr);
    return (getFPGAGateCategory(*block->quantumGate) & fpgaUnitaryPerm) == fpgaUnitaryPerm;
}

inline bool isSingleQubitNonCompBlock(const GateBlock* block) {
    assert(block != nullptr);
    return (getFPGAGateCategory(*block->quantumGate) & fpgaSingleQubitNonComp) == fpgaSingleQubitNonComp;
}

using tile_iter_t = std::list<std::array<GateBlock*, 36>>::iterator;


namespace {
GateBlock* computeCandidate(
        const GateBlock* lhs, const GateBlock* rhs, const FPGAFusionConfig& config) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;

    assert(lhs != rhs);
    assert(lhs->quantumGate != nullptr);
    assert(rhs->quantumGate != nullptr);

    // candidate block
    auto block = new GateBlock();

    // std::cerr << "Trying to fuse "
            //   << "lhs " << lhs->id << " and rhs " << rhs->id
            //   << " => candidate block " << block->id << "\n";

    std::vector<int> blockQubits;
    for (const auto& lData : lhs->dataVector) {
        const auto& q = lData.qubit;

        GateNode* lhsEntry = lData.lhsEntry;
        GateNode* rhsEntry;
        auto it = rhs->findQubit(q);
        if (it == rhs->dataVector.end())
            rhsEntry = lData.rhsEntry;
        else
            rhsEntry = it->rhsEntry;

        assert(lhsEntry);
        assert(rhsEntry);

        block->dataVector.push_back({q, lhsEntry, rhsEntry});
        blockQubits.push_back(q);
    }

    for (const auto& rData : rhs->dataVector) {
        const auto& q = rData.qubit;
        if (lhs->findQubit(q) == lhs->dataVector.end()) {
            block->dataVector.push_back(rData);
            blockQubits.push_back(q);
        }
    }

    // check fusion condition
    // 1. ignore single-qubit non-comp gates
    if (config.ignoreSingleQubitNonCompGates
            && (isSingleQubitNonCompBlock(lhs) || isSingleQubitNonCompBlock(rhs))) {
        std::cerr << CYAN_FG << "Omitted due to single-qubit non-comp gates\n" << RESET;
        return nullptr;
    }

    bool lhsIsUP = isUnitaryPermBlock(lhs);
    bool rhsIsUp = isUnitaryPermBlock(rhs);

    // 2. multi-qubit gates: only fuse when unitary perm
    if ((lhs->nqubits() > 1 || rhs->nqubits() > 1)
            && !(lhsIsUP && rhsIsUp)) {
        // std::cerr << CYAN_FG << "Rejected because there are multi-qubit gates, but not both of them are unitary perm\n" << RESET;
        return nullptr;
    }

    if ((lhsIsUP && rhsIsUp)
            && blockQubits.size() > config.maxUnitaryPermutationSize) {
        // std::cerr << CYAN_FG << "Rejecte because the candidate block size is too large\n" << RESET;
        return nullptr;
    }

    // accept candidate
    // std::cerr << GREEN_FG << "Fusion accepted!\n" << RESET;
    block->quantumGate = std::make_unique<QuantumGate>(
            rhs->quantumGate->lmatmul(*(lhs->quantumGate)));

    return block;
}
} // anonymous namespace

void saot::applyFPGAGateFusion(const FPGAFusionConfig& config, CircuitGraph& graph) {
    auto& tile = graph.tile();
    if (tile.size() < 2)
        return;
    
    bool hasChange = true;
    while (hasChange) {
        hasChange = false;
        auto tileIt = tile.begin();
        tile_iter_t tileNext;
        while ((tileNext = std::next(tileIt)) != tile.end()) {
            for (int q = 0; q < graph.nqubits; q++) {
                auto* lhs = (*tileIt)[q];
                auto* rhs = (*tileNext)[q];
                auto* candidate = computeCandidate(lhs, rhs, config);
                if (candidate) {
                    hasChange = true;
                    for (const auto& qq : lhs->getQubits())
                        (*tileIt)[qq] = nullptr;
                    for (const auto& qq : rhs->getQubits())
                        (*tileNext)[qq] = nullptr;
                    delete(lhs);
                    delete(rhs);
                    graph.insertBlock(tileIt, candidate);
                }
            }
            tileIt++;
        }
        graph.updateTileUpward();
        graph.eraseEmptyRows();
        if (!config.multiTraverse)
            break;
    }
}