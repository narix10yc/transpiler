#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "utils/iocolor.h"

using namespace IOColor;
using namespace saot;

FusionConfig FusionConfig::Disable = FusionConfig {
            .maxNQubits = 0,
            .maxOpCount = 0, 
            .zeroSkippingThreshold = 0.0,
            .allowMultipleTraverse = true,
            .incrementScheme = true
        };

FusionConfig FusionConfig::TwoQubitOnly = FusionConfig {
            .maxNQubits = 2,
            .maxOpCount = 64, // 2-qubit dense
            .zeroSkippingThreshold = 1e-8,
            .allowMultipleTraverse = true,
            .incrementScheme = true
        };

FusionConfig FusionConfig::Default = FusionConfig {
            .maxNQubits = 5,
            .maxOpCount = 256, // 3-qubit dense
            .zeroSkippingThreshold = 1e-8,
            .allowMultipleTraverse = true,
            .incrementScheme = true
        };

FusionConfig FusionConfig::Aggressive = FusionConfig {
            .maxNQubits = 7,
            .maxOpCount = 4096, // 5.5-qubit dense
            .zeroSkippingThreshold = 1e-8,
            .allowMultipleTraverse = true,
            .incrementScheme = true
        };

std::ostream& FusionConfig::display(std::ostream& OS) const {
    OS << CYAN_FG << "======== Fusion Config: ========\n" << RESET;
    OS << "max nqubits:          " << maxNQubits << "\n";
    OS << "max op count:         " << maxOpCount;
    if (maxOpCount < 0)
        OS << " (infinite)";
    OS << "\n";
    OS << "zero skip thres:      " << std::scientific << zeroSkippingThreshold << "\n";
    OS << "allow multi traverse: " << ((allowMultipleTraverse) ? "true" : "false") << "\n";
    OS << "increment scheme:     " << ((incrementScheme) ? "true" : "false") << "\n";
    OS << CYAN_FG << "================================\n" << RESET;
    return OS;
}

// convenient iterator types
using tile_iter_t = std::list<std::array<GateBlock*, 36>>::iterator;

GateBlock* computeCandidate(
        const GateBlock* lhs, const GateBlock* rhs, const FusionConfig& config) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;
    
    if (lhs == rhs)
        return nullptr;
    
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
    
    block->nqubits = block->dataVector.size();

    // check fusion condition
    if (block->nqubits > config.maxNQubits) {
        // std::cerr << CYAN_FG << "Rejected due to maxNQubits\n" << RESET;
        delete(block);
        return nullptr;
    }
    
    auto blockQuantumGate = rhs->quantumGate->lmatmul(*(lhs->quantumGate));
    if (config.maxNQubits >= 0 && 
            blockQuantumGate.opCount(config.zeroSkippingThreshold) > config.maxOpCount) {
        // std::cerr << CYAN_FG << "Rejected due to OpCount\n" << RESET;
        delete(block);
        return nullptr;
    }

    // accept candidate
    // std::cerr << GREEN_FG << "Fusion accepted!\n" << RESET;
    block->quantumGate = std::make_unique<QuantumGate>(blockQuantumGate);
    return block;
}

GateBlock* trySameWireFuse(const FusionConfig& config, CircuitGraph& graph,
        const tile_iter_t& itLHS, int q_) {
    assert(itLHS != graph.tile().end());
    const auto itRHS = std::next(itLHS);
    if (itRHS == graph.tile().end())
        return nullptr;

    GateBlock* lhs = (*itLHS)[q_];
    GateBlock* rhs = (*itRHS)[q_];

    if (!lhs || !rhs)
        return nullptr;
  
    GateBlock* block = computeCandidate(lhs, rhs, config);
    if (block == nullptr)
        return nullptr;

    // std::cerr << BLUE_FG;
    // lhs->displayInfo(std::cerr);
    // rhs->displayInfo(std::cerr);
    // block->displayInfo(std::cerr) << RESET;

    for (const auto& q : lhs->getQubits())
        (*itLHS)[q] = nullptr;
    for (const auto& q : rhs->getQubits())
        (*itRHS)[q] = nullptr;

    delete(lhs);
    delete(rhs);

    // insert block to tile
    graph.insertBlock(itLHS, block);
    return block;
}

GateBlock* tryCrossWireFuse(const FusionConfig& config, CircuitGraph& graph,
        const tile_iter_t& tileIt, int q) {
    auto block0 = (*tileIt)[q];
    if (block0 == nullptr)
        return nullptr;
    
    for (unsigned q1 = 0; q1 < graph.nqubits; q1++) {
        auto* block1 = (*tileIt)[q1];
        auto* fusedBlock = computeCandidate(block0, block1, config);
        if (fusedBlock == nullptr)
            continue;
        for (const auto q : fusedBlock->getQubits()) {
            (*tileIt)[q] = fusedBlock;
        }
        delete(block0);
        delete(block1);
        return fusedBlock;
    }
    return nullptr;
}

void saot::applyGateFusion(const FusionConfig& originalConfig, CircuitGraph& graph) {
    auto& tile = graph.tile();
    if (tile.size() < 2)
        return;

    GateBlock* lhsBlock;
    GateBlock* rhsBlock;

    auto config = originalConfig;
    // increment scheme applies maxNQubits = 2, 3, ..., maxNQubits
    if (config.incrementScheme)
        config.maxNQubits = 2;

    do {
        bool hasChange = true;
        tile_iter_t tileIt;
        unsigned q = 0;
        // multi-traversal
        while (hasChange) {
            tileIt = tile.begin();
            hasChange = false;
            while (std::next(tileIt) != tile.end()) {
                // same wire (connected consecutive) fuse
                q = 0;
                while (q < graph.nqubits) {
                    if ((*tileIt)[q] == nullptr) {
                        q++;
                        continue;
                    }
                    if ((*std::next(tileIt))[q] == nullptr) {
                        graph.repositionBlockDownward(tileIt, q++);
                        continue;
                    }
                    auto* fusedBlock = trySameWireFuse(config, graph, tileIt, q);
                    if (fusedBlock == nullptr)
                        q++;
                    else
                        hasChange = true;
                }
                // cross wire (same row) fuse
                q = 0;
                while (q < graph.nqubits) {
                    auto* fusedBlock = tryCrossWireFuse(config, graph, tileIt, q);
                    if (fusedBlock == nullptr)
                        q++;
                    else
                        hasChange = true;
                }
                tileIt++;
            }
            graph.eraseEmptyRows();
            // graph.updateTileUpward();
            if (!config.allowMultipleTraverse)
                break;
        }
    } while (++(config.maxNQubits) <= originalConfig.maxNQubits);
}