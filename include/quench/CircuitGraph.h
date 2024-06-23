#ifndef QUENCH_CIRCUITGRAPH_H
#define QUENCH_CIRCUITGRAPH_H

#include <vector>
#include <array>
#include <set>
#include <list>
#include <functional>
#include "quench/ast.h"

namespace quench::circuit_graph {

class GateNode {
public:
    struct gate_data {
        unsigned qubit;
        GateNode* lhsGate;
        GateNode* rhsGate;
    };
    unsigned nqubits;
    cas::GateMatrix gateMatrix;
    std::vector<gate_data> dataVector;

    GateNode(const cas::GateMatrix& gateMatrix,
             const std::vector<unsigned>& qubits)
        : nqubits(gateMatrix.nqubits),
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
};

class GateBlock {
public:
    struct block_data {
        unsigned qubit;
        GateNode* lhsEntry;
        GateNode* rhsEntry;
    };

    const int id;
    unsigned nqubits;
    std::vector<block_data> dataVector;

    GateBlock(int id) : id(id), nqubits(0), dataVector() {}

    GateBlock(int id, GateNode* gate)
        : id(id), nqubits(gate->nqubits), dataVector()
    {
        for (const auto& data : gate->dataVector)
            dataVector.push_back({data.qubit, gate, gate});
    }

    std::ostream& displayInfo(std::ostream& os) const;

    size_t countGates() const;

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

    void applyInOrder(std::function<void(GateNode*)>) const;
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

    void eraseEmptyRows();

    void repositionBlockUpward(tile_iter_t it, size_t q_);

    void repositionBlockDownward(tile_riter_t it, size_t q_);
    void repositionBlockDownward(tile_iter_t it, size_t q_) {
        return repositionBlockDownward(--std::make_reverse_iterator(it), q_);
    }

    void updateTileUpward();
    void updateTileDownward();

    /// @brief 
    /// @param it 
    /// @param q_ 
    /// @return -1000 if it is at the last row; -100 if block is null; 
    /// Otherwise, return the number of qubits after fusion 
    int checkFuseCondition(tile_const_iter_t it, size_t q_) const;
    GateBlock* fuse(tile_iter_t tileLHS, size_t q);
public:
    unsigned nqubits;

    CircuitGraph()
        : currentBlockId(0), tile(1, {nullptr}), nqubits(0) {}

    void addGate(const cas::GateMatrix& matrix,
                 const std::vector<unsigned>& qubits);

    std::set<GateBlock*> getAllBlocks() const;

    size_t countGates() const;
    size_t countBlocks() const;

    std::ostream& print(std::ostream& os, int verbose = 1) const;
    std::ostream& displayInfo(std::ostream& os, int verbose = 1) const;

    void dependencyAnalysis();

    void fuseToTwoQubitGates();

    void greedyGateFusion(int maxNQubits);

    void applyInOrder(std::function<void(GateBlock*)>) const;

};


}; // namespace quench::circuit_graph

#endif // QUENCH_CIRCUITGRAPH_H