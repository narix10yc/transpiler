#ifndef SAOT_CIRCUITGRAPH_H
#define SAOT_CIRCUITGRAPH_H

#include "saot/QuantumGate.h"

#include <vector>
#include <array>
#include <set>
#include <list>
#include <memory>

namespace saot {

class GateNode {
private:
    static int idCount;
public:
    struct gate_data {
        int qubit;
        GateNode* lhsGate;
        GateNode* rhsGate;
    };
    const int id;
    unsigned nqubits;
    GateMatrix gateMatrix;
    std::vector<gate_data> dataVector;

    GateNode(const GateMatrix& gateMatrix, const std::vector<int>& qubits)
        : id(idCount++),
          nqubits(gateMatrix.nqubits()),
          gateMatrix(gateMatrix),
          dataVector(gateMatrix.nqubits()) {
        assert(gateMatrix.nqubits() == qubits.size());
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

    std::vector<int> getQubits() const {
        std::vector<int> qubits(nqubits);
        for (unsigned i = 0; i < nqubits; i++)
            qubits[i] = dataVector[i].qubit;
        
        return qubits;
    }

    QuantumGate toQuantumGate() const {
        return QuantumGate(gateMatrix, getQubits());
    }
};

class GateBlock {
private:
    static int idCount;
public:
    struct block_data {
        int qubit;
        GateNode* lhsEntry;
        GateNode* rhsEntry;
    };

    int id;
    std::vector<block_data> dataVector;
    std::unique_ptr<QuantumGate> quantumGate;

    GateBlock() : id(idCount++), dataVector(), quantumGate(nullptr) {}

    GateBlock(GateNode* gateNode)
           : id(idCount++), dataVector(),
             quantumGate(std::make_unique<QuantumGate>(gateNode->toQuantumGate())) {
        for (const auto& data : gateNode->dataVector)
            dataVector.push_back({data.qubit, gateNode, gateNode});
    }

    std::ostream& displayInfo(std::ostream& os) const;

    std::vector<GateNode*> getOrderedGates() const;

    size_t countGates() const { return getOrderedGates().size(); }

    int connect(GateBlock* rhsBlock, int q = -1);

    int nqubits() const { return dataVector.size(); }
    
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
        auto it = dataVector.cbegin();
        while (it != dataVector.cend()) {
            if (it->qubit == q)
                break;
            it++;
        }
        return it;
    }

    bool hasSameTargets(const GateBlock& other) const {
        if (nqubits() != other.nqubits())
            return false;
        for (const auto& data : other.dataVector) {
            if (findQubit(data.qubit) == dataVector.end())
                return false;
        }
        return true;
    }

    // TODO: This should be identical to quantumGate->qubits
    // Find a way to remove the redundency
    std::vector<int> getQubits() const {
        std::vector<int> vec(dataVector.size());
        for (unsigned i = 0; i < dataVector.size(); i++)
            vec[i] = dataVector[i].qubit;
        return vec;
    }

    void internalFuse() {
        assert(false && "Not Implemented");
    }
};

class CircuitGraph {
private:
    using row_t = std::array<GateBlock*, 36>;
    using tile_t = std::list<row_t>;
    using tile_iter_t = std::list<row_t>::iterator;
    using tile_riter_t = std::list<row_t>::reverse_iterator;
    using tile_const_iter_t = std::list<row_t>::const_iterator;
    tile_t _tile;

public:
    int nqubits;

    CircuitGraph()
        : _tile(1, {nullptr}), nqubits(0) {}

    static CircuitGraph QFTCircuit(int nqubits);
    static CircuitGraph ALACircuit(int nqubits, int nrounds);

    tile_t& tile() { return _tile; }
    const tile_t& tile() const { return _tile; }

    /// @brief Erase empty rows in the tile
    void eraseEmptyRows();

    bool isRowVacant(tile_iter_t it, const GateBlock* block) const {
        for (const auto& q : block->getQubits())
            if ((*it)[q] != nullptr)
                return false;
        return true;
    }

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

    /// @brief Try to insert block to a specified row. Three outcome may happen:
    /// - If \p it is vacant, insert \p block there. Otherwise,
    /// - If \p it+1 is vacant, insert \p block there. Otherwise,
    /// - Insert a separate row between \p it and \p it+1 and place \p block
    ///   there.
    tile_iter_t insertBlock(tile_iter_t it, GateBlock* block);

    void addGate(const QuantumGate& gate) {
        return addGate(gate.gateMatrix, gate.qubits);
    }

    void addGate(const GateMatrix& matrix,
                 const std::vector<int>& qubits);

    /// @return ordered vector of blocks
    std::vector<GateBlock*> getAllBlocks() const;

    /// @brief Get the number of blocks with each size.
    /// @return ret[i] is the number of blocks with size i. Therefore, ret[0] is 
    /// always 0, and ret.size() == largest_size + 1.
    std::vector<int> getBlockSizes() const;

    std::vector<std::vector<int>> getBlockOpCountHistogram() const;

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
};


}; // namespace saot

#endif // SAOT_CIRCUITGRAPH_H