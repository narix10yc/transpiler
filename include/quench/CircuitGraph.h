#ifndef QUENCH_CIRCUITGRAPH_H
#define QUENCH_CIRCUITGRAPH_H

#include <vector>
#include <array>
#include <set>
#include <map>
#include <cassert>
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

    GateNode(const cas::GateMatrix& matrix)
        : nqubits(matrix.nqubits), matrix(matrix), dataVector() {}

    void updateLHS(GateNode* newGate, unsigned q) {
        for (auto& data : dataVector) {
            if (data.qubit == q) {
                data.lhsGate = newGate;
                return;
            }
        }
        assert(false && "q is not in qubits");
    }

    void updateRHS(GateNode* newGate, unsigned q) {
        for (auto& data : dataVector) {
            if (data.qubit == q) {
                data.rhsGate = newGate;
                return;
            }
        }
        assert(false && "q is not in qubits");
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

    GateBlock(int id) : id(id), nqubits(0), dataVector() {}

    GateBlock(int id, GateNode* gate)
        : id(id), nqubits(gate->nqubits), dataVector(gate->nqubits)
    {
        for (const auto& data : gate->dataVector)
            dataVector.push_back({data.qubit, gate, gate});
    }
};

class CircuitGraph {
public:
    std::vector<std::array<GateBlock*, 36>> tile;
    std::array<GateNode*, 36> lhsEntry, rhsEntry;
    int currentBlockId;
    unsigned nqubits;

    CircuitGraph()
        : tile(), lhsEntry({}), rhsEntry({}), currentBlockId(0), nqubits(0) {}

    void addGate(const cas::GateMatrix& matrix,
                      const std::vector<unsigned>& qubits);

    std::ostream& print(std::ostream& os) const;

    void dependencyAnalysis();

    void fuseToTwoQubitGates();

    void greedyGateFusion();


};


}; // namespace quench::circuit_graph

#endif // QUENCH_CIRCUITGRAPH_H