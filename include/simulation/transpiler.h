#ifndef SIMULATION_TRANSPILER_H
#define SIMULATION_TRANSPILER_H

#include "qch/ast.h"
#include "simulation/types.h"

#include <vector>
#include <set>
#include <unordered_set>
#include <iostream>

namespace simulation::transpile {
class GateNode {
public:
    struct node_data {
        unsigned qubit;
        GateNode* leftNode = nullptr;
        GateNode* rightNode = nullptr;
    };
    unsigned nqubits;
    SquareComplexMatrix<double> matrix;
    std::vector<node_data> dataVector;
    unsigned id;
    GateNode(unsigned nqubits, unsigned id)
        : nqubits(nqubits), matrix(1<<nqubits),
          dataVector(nqubits), id(id) {}

    bool actOnSameQubits(const GateNode& other) {
        if (nqubits != other.nqubits)
            return false;
        
        for (auto& data : dataVector) {
            auto q = data.qubit;
            bool flag = false;
            for (auto& otherData : other.dataVector) {
                if (q == otherData.qubit) {
                    flag = true;
                    break;
                }
            }
            if (flag)
                continue;
            return false;
        }
        return true;
    }

    node_data& operator[](size_t idx) {
        assert(idx < dataVector.size());
        return dataVector[idx];
    }

    const node_data& operator[](size_t idx) const {
        assert(idx < dataVector.size());
        return dataVector[idx];
    }

    bool actsOnQubit(unsigned q) const {
        for (auto& data : dataVector) {
            if (data.qubit == q)
                return true;
        }
        return false;
    }

    std::vector<unsigned> qubits() const {
        std::vector<unsigned> arr(nqubits);
        for (unsigned i = 0; i < nqubits; i++)
            arr[i] = dataVector[i].qubit;
        return arr;
    }

}; // class GateNode

class CircuitGraph {
    unsigned count;
    struct CompareGateNodePointers {
        bool operator()(const GateNode* lhs, const GateNode* rhs) const {
            return lhs->id < rhs->id;
        }
    };
public:
    std::vector<GateNode*> leftEntry;
    std::vector<GateNode*> rightEntry;

    /// @brief The collection of allNodes will be sorted by their id. Their id
    /// also tells the order of gates
    std::set<GateNode*, CompareGateNodePointers> allNodes;
private:
    /// @brief Try connect two nodes
    /// @param q: Optional. Specify which qubit to connect on. When set to <0,
    ///     connect along all possible qubits
    /// @return Number of connections
    unsigned connectTwoNodes(GateNode* left, GateNode* right, int q=-1);

    /// @brief Try absorb all directly connected single-qubit gates.
    /// @param node can either be single-qubit or two-qubit.
    /// @return Number of gates absorbed. 
    unsigned absorbNeighbouringSingleQubitGates(GateNode* node);
    unsigned absorbNeighbouringTwoQubitGates(GateNode* node);

public:
    CircuitGraph() : count(0), leftEntry(32, nullptr),
                     rightEntry(32, nullptr), allNodes() {}

    static CircuitGraph FromQch(const qch::ast::RootNode& root);

    qch::ast::RootNode toQch() const;

    void removeGateNode(GateNode* node);

    /// @brief Add a single-qubit gate into the right-most of the graph.
    /// Connections will be adjusted accordingly.
    /// @param u3 A U3Gate
    /// @return The node
    GateNode* addSingleQubitGate(const U3Gate& u3);

    GateNode* addTwoQubitGate(const U2qGate& u2q);

    void transpileForCPU();

    std::vector<GateNode> getNodesInOrder() const;

    bool sanityCheck(std::ostream& os) const;

    void draw(std::ostream& os) const;

}; // class CircuitGraph

} // namespace simulation

#endif // SIMULATION_TRANSPILER_H