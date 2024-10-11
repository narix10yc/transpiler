#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/QuantumGate.h"

using namespace saot;

FPGAFusionConfig FPGAFusionConfig::Default = FPGAFusionConfig {
            .maxUnitaryPermutationSize = 5,
            .ignoreSingleQubitNonCompGates = true,
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
        default: break; 
    }

    // TODO: handle general gates
    return fpgaGeneral;
}

inline bool isUnitaryPermBlock(const GateBlock* block) {
    assert(block != nullptr);
    return getFPGAGateCategory(*block->quantumGate) & fpgaUnitaryPerm;
}

inline bool isSingleQubitNonCompBlock(const GateBlock* block) {
    assert(block != nullptr);
    return getFPGAGateCategory(*block->quantumGate) & fpgaSingleQubitNonComp;
}

using tile_iter_t = std::list<std::array<GateBlock*, 36>>::iterator;

GateBlock* tryFuse(const FPGAFusionConfig& config, CircuitGraph& graph,
        const tile_iter_t& itLHS, int q0) {
    GateBlock* lhs = (*itLHS)[q0];
    if (lhs == nullptr)
        return nullptr;
    
    auto itRHS = std::next(itLHS);
    if (itRHS == graph.tile().end())
        return nullptr;

    GateBlock* rhs = (*itRHS)[q0];
    if (rhs == nullptr)
        return nullptr;

    // eliminate un-fuseable cases
    // 1. ignore single-qubit non-comp gates
    if (config.ignoreSingleQubitNonCompGates
        && (isSingleQubitNonCompBlock(lhs) || isSingleQubitNonCompBlock(rhs)))
        return nullptr;
    
    // 2. multi-qubit gates: only fuse when unitary perm
    if ((lhs->nqubits > 1 || rhs->nqubits > 1)
        && !(isUnitaryPermBlock(lhs) && isUnitaryPermBlock(rhs)))
        return nullptr;
    
    auto* fusedBlock = new GateBlock(std::make_unique<QuantumGate>(
        rhs->quantumGate->lmatmul(*lhs->quantumGate)));

    for (const auto& q : lhs->getQubits())
        (*itLHS)[q] = nullptr;
    for (const auto& q : rhs->getQubits())
        (*itRHS)[q] = nullptr;
    delete(lhs);
    delete(rhs);

    graph.insertBlock(itRHS, fusedBlock);
    return fusedBlock;
}

void saot::applyFPGAGateFusion(const FPGAFusionConfig& config, CircuitGraph& graph) {
    auto& tile = graph.tile();
    if (tile.size() < 2)
        return;

    for (auto tileIt = tile.begin(); tileIt != tile.end(); tileIt++) {
        for (int q = 0; q < graph.nqubits; q++)
            tryFuse(config, graph, tileIt, q);
    }
}