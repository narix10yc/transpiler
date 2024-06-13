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

class GateChain {
public:
    struct chain_data {
        unsigned qubit;
        GateNode* lhsEntry;
        GateNode* rhsEntry;
    };

    int id;
    unsigned nqubits;
    std::vector<chain_data> dataVector;

    GateChain(int id) : id(id), nqubits(0), dataVector() {}

    GateChain(int id, GateNode* gate)
        : id(id), nqubits(gate->nqubits), dataVector(gate->nqubits)
    {
        for (const auto& data : gate->dataVector)
            dataVector.push_back({data.qubit, gate, gate});
    }
};

class CircuitGraph {
public:
    std::vector<std::array<GateChain*, 36>> tile;
    std::array<GateNode*, 36> lhsEntry, rhsEntry;
    int currentChainId;
    unsigned nqubits;

    CircuitGraph()
        : tile(), lhsEntry({}), rhsEntry({}), currentChainId(0), nqubits(0) {}

    void addGate(const cas::GateMatrix& matrix,
                      const std::vector<unsigned>& qubits);

    std::ostream& print(std::ostream& os) const;

    void updateChains();


};


}; // namespace quench::circuit_graph

#endif // QUENCH_CIRCUITGRAPH_H