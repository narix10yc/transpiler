#ifndef QUENCH_CIRCUITGRAPH_H
#define QUENCH_CIRCUITGRAPH_H

#include <vector>
#include <array>
#include <set>
#include <map>
#include <cassert>
#include <algorithm>
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
    cas::GateMatrix matrix;
    std::vector<gate_data> dataVector;

    GateNode(const cas::GateMatrix& matrix, const std::vector<unsigned>& qubits)
        : nqubits(matrix.nqubits), matrix(matrix), dataVector(matrix.nqubits) {
            assert(matrix.nqubits == qubits.size());
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

    int connect(GateNode* rhsGate, int q = -1) {
        assert(rhsGate != nullptr);

        if (q >= 0) {
            auto myIt = findQubit(static_cast<unsigned>(q));
            if (myIt == dataVector.end())
                return 0;
            auto rhsIt = rhsGate->findQubit(static_cast<unsigned>(q));
            if (rhsIt == rhsGate->dataVector.end())
                return 0;
            
            myIt->rhsGate = rhsGate;
            rhsIt->lhsGate = this;
            return 1;
        }
        int count = 0;
        for (auto& data : dataVector) {
            auto rhsIt = rhsGate->findQubit(data.qubit);
            if (rhsIt == rhsGate->dataVector.end())
                continue;
            data.rhsGate = rhsGate;
            rhsIt->lhsGate = this;
        }
        return count;
    }
};

class GateBlock {
public:
    struct block_data {
        unsigned qubit;
        GateNode* lhsEntry;
        GateNode* rhsEntry;
        GateBlock* lhsBlock;
        GateBlock* rhsBlock;
    };

    int id;
    int lineNumber;
    unsigned nqubits;
    std::vector<block_data> dataVector;
    // line number in the tile

    GateBlock(int id) : id(id), lineNumber(-1), nqubits(0), dataVector() {}

    GateBlock(int id, GateNode* gate)
        : id(id), lineNumber(-1), nqubits(gate->nqubits), dataVector()
    {
        for (const auto& data : gate->dataVector)
            dataVector.push_back({data.qubit, gate, gate, nullptr, nullptr});
    }

    ~GateBlock() {
        std::cerr << "deleting block " << id << "\n";
    }

    std::vector<block_data>::iterator findQubit(unsigned q) {
        auto it = dataVector.begin();
        while (it != dataVector.end()) {
            if (it->qubit == q)
                break;
            it++;
        }
        return it;
    }

    int connect(GateBlock* rhsGate, int q = -1) {
        assert(rhsGate != nullptr);

        if (q >= 0) {
            auto myIt = findQubit(static_cast<unsigned>(q));
            if (myIt == dataVector.end())
                return 0;
            auto rhsIt = rhsGate->findQubit(static_cast<unsigned>(q));
            if (rhsIt == rhsGate->dataVector.end())
                return 0;
            
            myIt->rhsBlock = rhsGate;
            rhsIt->lhsBlock = this;
            return 1;
        }
        int count = 0;
        for (auto& data : dataVector) {
            auto rhsIt = rhsGate->findQubit(data.qubit);
            if (rhsIt == rhsGate->dataVector.end())
                continue;
            data.rhsBlock = rhsGate;
            rhsIt->lhsBlock = this;
        }
        return count;
    }

    void fuseWithRHS(GateBlock* rhsBlock);

};

class CircuitGraph {
private:
    int currentBlockId;
    struct block_ptr_cmp_by_id {
        bool operator()(const GateBlock* a, const GateBlock* b) const {
            return a->id < b->id;
        }
    };
public:
    std::set<GateBlock*, block_ptr_cmp_by_id> allBlocks;
    std::array<GateBlock*, 36> lhsEntry, rhsEntry;
    unsigned nqubits;

    CircuitGraph()
        : currentBlockId(0), allBlocks(),
          lhsEntry({}), rhsEntry({}), nqubits(0) {}

    void addGate(const cas::GateMatrix& matrix,
                      const std::vector<unsigned>& qubits);

    GateBlock* createBlock(GateNode* gate);
    
    void destroyBlock(GateBlock* block) {
        assert(block != nullptr);
        assert(allBlocks.find(block) != allBlocks.end());

        std::cerr << "destroying block " << block->id << "\n";
        for (const auto& data : block->dataVector) {
            if (data.lhsBlock && data.rhsBlock) {
                data.lhsBlock->connect(data.rhsBlock, data.qubit);
                std::cerr << "reconnected " << data.lhsBlock->id << " and "
                          << data.rhsBlock->id << "\n";
            }
        }

        allBlocks.erase(block);
        // delete(block);
        std::cerr << "destroyed " << block->id << "\n";
    }

    std::ostream& print(std::ostream& os) const;

    void dependencyAnalysis();

    void fuseToTwoQubitGates();

    void greedyGateFusion();

};


}; // namespace quench::circuit_graph

#endif // QUENCH_CIRCUITGRAPH_H