#ifndef QUENCH_CIRCUITGRAPH_H
#define QUENCH_CIRCUITGRAPH_H

#include <vector>
#include <array>
#include <set>
#include <list>
#include <functional>
#include "quench/ast.h"
#include "quench/QuantumGate.h"

namespace quench::circuit_graph {

using QuantumGate = quench::quantum_gate::QuantumGate;

class GateNode {
public:
    struct gate_data {
        unsigned qubit;
        GateNode* lhsGate;
        GateNode* rhsGate;
    };
    const int id;
    unsigned nqubits;
    quantum_gate::GateMatrix gateMatrix;
    std::vector<gate_data> dataVector;

    GateNode(int id,
             const quantum_gate::GateMatrix& gateMatrix,
             const std::vector<unsigned>& qubits)
        : id(id),
          nqubits(gateMatrix.nqubits),
          gateMatrix(gateMatrix),
          dataVector(gateMatrix.nqubits)
    {
        assert(gateMatrix.nqubits == qubits.size());
        for (unsigned i = 0; i < qubits.size(); i++)
            dataVector[i] = { qubits[i], nullptr, nullptr };
    }

    std::vector<gate_data>::iterator findQubit(unsigned q) {
        auto it = dataVector.begin();
        while (it != dataVector.end()) {
            if (it->qubit == q)
                break;
            it++;
        }
        return it;
    }

    GateNode* findLHS(unsigned q) {
        for (const auto& data : dataVector) {
            if (data.qubit == q)
                return data.lhsGate;
        }
        return nullptr;
    }

    GateNode* findRHS(unsigned q) {
        for (const auto& data : dataVector) {
            if (data.qubit == q)
                return data.rhsGate;
        }
        return nullptr;
    }

    int connect(GateNode* rhsGate, int q = -1);

    std::vector<unsigned> getQubits() const {
        std::vector<unsigned> qubits(nqubits);
        for (unsigned i = 0; i < nqubits; i++)
            qubits[i] = dataVector[i].qubit;
        
        return qubits;
    }

    quantum_gate::QuantumGate toQuantumGate() const {
        return QuantumGate(gateMatrix, getQubits());
    }
};

class GateBlock {
public:
    struct block_data {
        unsigned qubit;
        GateNode* lhsEntry;
        GateNode* rhsEntry;
    };

    int id;
    unsigned nqubits;
    std::vector<block_data> dataVector;
    std::unique_ptr<QuantumGate> quantumGate;

    GateBlock(int id, std::unique_ptr<QuantumGate> quantumGate = nullptr)
        : id(id), nqubits(0), dataVector(),
          quantumGate(std::move(quantumGate)) {}

    GateBlock(int id, GateNode* gateNode)
        : id(id), nqubits(gateNode->nqubits), dataVector(),
          quantumGate(std::make_unique<QuantumGate>(gateNode->toQuantumGate()))
    {
        for (const auto& data : gateNode->dataVector)
            dataVector.push_back({data.qubit, gateNode, gateNode});
    }

    std::ostream& displayInfo(std::ostream& os) const;

    std::vector<GateNode*> getOrderedGates() const;

    size_t countGates() const {
        return getOrderedGates().size();
    }

    int connect(GateBlock* rhsBlock, int q = -1);

    std::vector<block_data>::iterator findQubit(unsigned q) {
        auto it = dataVector.begin();
        while (it != dataVector.end()) {
            if (it->qubit == q)
                break;
            it++;
        }
        return it;
    }

    std::vector<block_data>::const_iterator findQubit(unsigned q) const {
        auto it = dataVector.begin();
        while (it != dataVector.end()) {
            if (it->qubit == q)
                break;
            it++;
        }
        return it;
    }

    bool hasSameTargets(const GateBlock& other) const {
        if (nqubits != other.nqubits)
            return false;
        for (const auto& data : other.dataVector) {
            if (findQubit(data.qubit) == dataVector.end())
                return false;
        }
        return true;
    }

    std::vector<int> getQubits() const {
        std::vector<int> vec(dataVector.size());
        for (unsigned i = 0; i < dataVector.size(); i++)
            vec[i] = dataVector[i].qubit;
        return vec;
    }
};

struct FusionConfig {
    int maxNQubits;
    int maxOpCount;
    double zeroSkippingThreshold;
public:
    static FusionConfig Disable() {
        return {
            .maxNQubits = 0,
            .maxOpCount = 0, 
            .zeroSkippingThreshold = 0.0
        };
    }

    static FusionConfig Default() {
        return {
            .maxNQubits = 4,
            .maxOpCount = 128, // 3-qubit dense
            .zeroSkippingThreshold = 1e-8
        };
    }

    static FusionConfig TwoQubitOnly() {
        return {
            .maxNQubits = 2,
            .maxOpCount = 32, // 2-qubit dense
            .zeroSkippingThreshold = 1e-8
        };
    }

    static FusionConfig Aggressive() {
        return {
            .maxNQubits = 5,
            .maxOpCount = 1024, // 4-qubit dense takes 512 op
            .zeroSkippingThreshold = 1e-8
        };
    }

    static FusionConfig Preset(int level) {
        if (level == 0)
            return FusionConfig::Disable();
        if (level == 1) 
            return FusionConfig::TwoQubitOnly();
        if (level == 2) 
            return FusionConfig::Default();
        if (level == 3)
            return FusionConfig::Aggressive();
        assert(false && "Unsupported FusionConfig preset");
        return FusionConfig::Disable();
    }
};

class CircuitGraph {
private:
    using row_t = std::array<GateBlock*, 36>;
    using tile_t = std::list<row_t>;
    using tile_iter_t = std::list<row_t>::iterator;
    using tile_riter_t = std::list<row_t>::reverse_iterator;
    using tile_const_iter_t = std::list<row_t>::const_iterator;
    int currentBlockId;
    tile_t tile;
    FusionConfig fusionConfig;

    /// @brief Erase empty rows in the tile
    void eraseEmptyRows();

    tile_iter_t repositionBlockUpward(tile_iter_t it, size_t q_);
    tile_iter_t repositionBlockUpward(tile_riter_t it, size_t q_) {
        return repositionBlockUpward(--(it.base()), q_);
    }

    tile_riter_t repositionBlockDownward(tile_riter_t it, size_t q_);
    tile_riter_t repositionBlockDownward(tile_iter_t it, size_t q_) {
        return repositionBlockDownward(--std::make_reverse_iterator(it), q_);
    }

    void updateTileUpward();
    void updateTileDownward();

    GateBlock* fusionCandidate(GateBlock* lhs, GateBlock* rhs);

    tile_iter_t insertBlock(tile_iter_t it, GateBlock* block);

    GateBlock* tryFuseConnectedConsecutive(tile_iter_t tileLHS, size_t q);

    GateBlock* tryFuseSameRow(tile_iter_t tileIt, size_t q);

public:
    unsigned nqubits;

    CircuitGraph(FusionConfig fusionConfig = FusionConfig::Disable())
        : currentBlockId(0), tile(1, {nullptr}),
          nqubits(0), fusionConfig(fusionConfig) {}

    void updateFusionConfig(const FusionConfig& newConfig) {
        fusionConfig = newConfig;
    }

    void addGate(const quantum_gate::GateMatrix& matrix,
                 const std::vector<unsigned>& qubits);

    /// @return ordered vector of blocks
    std::vector<GateBlock*> getAllBlocks() const;

    /// @brief Get the number of blocks with each size.
    /// @return ret[i] is the number of blocks with size i. Therefore, ret[0] is 
    /// always 0, and ret.size() == largest_size + 1.
    std::vector<int> getBlockSizes() const;

    size_t countBlocks() const {
        return getAllBlocks().size();
    }

    size_t countGates() const {
        const auto allBlocks = getAllBlocks();
        size_t sum = 0;
        for (const auto& block : allBlocks)
            sum += block->countGates();
        return sum;
    }

    size_t countTotalOps() const {
        const auto allBlocks = getAllBlocks();
        size_t sum = 0;
        for (const auto& block : allBlocks) {
            assert(block->quantumGate != nullptr);
            sum += block->quantumGate->opCount();
        }
        return sum;
    }

    void relabelBlocks();

    /// @brief Console print the tile.
    /// @param verbose If > 1, also print the address of each row in front
    std::ostream& print(std::ostream& os = std::cerr, int verbose = 1) const;

    std::ostream& displayInfo(std::ostream& os = std::cerr, int verbose = 1) const;

    std::ostream& displayFusionConfig(std::ostream& os = std::cerr) const;

    FusionConfig& getFusionConfig() { return fusionConfig; }

    void dependencyAnalysis();

    void greedyGateFusion();

};


}; // namespace quench::circuit_graph

#endif // QUENCH_CIRCUITGRAPH_H