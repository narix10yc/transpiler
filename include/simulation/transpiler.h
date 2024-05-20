#ifndef SIMULATION_TRANSPILER_H
#define SIMULATION_TRANSPILER_H

#include "qch/ast.h"
#include "simulation/types.h"

#include <vector>
#include <set>
#include <unordered_set>

namespace simulation::transpile {
class GateNode {
public:
    unsigned nqubits;
    SquareComplexMatrix<double> matrix;
    std::vector<unsigned> qubits;
    std::vector<GateNode*> leftNodes, rightNodes;

    GateNode(unsigned nqubits) : nqubits(nqubits),
                                 matrix(1<<nqubits),
                                 qubits(nqubits),
                                 leftNodes(nqubits, nullptr),
                                 rightNodes(nqubits, nullptr) {}

    bool actOnSameQubits(const GateNode& other) {
        if (nqubits != other.nqubits)
            return false;
        
        std::multiset<unsigned> set1(qubits.begin(), qubits.end());
        std::multiset<unsigned> set2(other.qubits.begin(), other.qubits.end());

        return set1 == set2;
    }

}; // class GateNode

class CircuitGraph {
public:
    std::vector<GateNode*> leftEntry;
    std::vector<GateNode*> rightEntry;

    std::unordered_set<GateNode*> allNodes;

private:
    /// @brief Connect two nodes along qubit q.
    /// @return true if both left and right have qubit q.
    bool connectTwoNodes(GateNode* left, GateNode* right, unsigned q);

    /// @brief Try absorb all directly connected single-qubit gates.
    /// @param node can either be single-qubit or two-qubit.
    /// @return Number of gates absorbed. 
    unsigned absorbNeighbouringSingleQubitGates(GateNode* node);
    unsigned absorbNeighbouringTwoQubitGates(GateNode* node);

public:
    CircuitGraph() : leftEntry(32, nullptr), rightEntry(32, nullptr), allNodes() {}

    static CircuitGraph FromQch(const qch::ast::RootNode& root);

    qch::ast::RootNode toQch() const;

    GateNode* addGateNode(unsigned nqubits);

    void removeGateNode(GateNode* node);

    void addSingleQubitGate(const U3Gate& u3);

    void addTwoQubitGate(const U2qGate& u2q);

    void transpileForCPU();

}; // class CircuitGraph

} // namespace simulation

#endif // SIMULATION_TRANSPILER_H