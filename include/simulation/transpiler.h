#ifndef SIMULATION_TRANSPILER_H
#define SIMULATION_TRANSPILER_H

#include "qch/ast.h"
#include "simulation/types.h"

#include <vector>
#include <stdexcept>

namespace simulation::transpile {

class CircuitGraph {
    class GateNode {
    public:
        unsigned nqubits;
        std::vector<double> matrixReal, matrixImag;
        std::vector<unsigned> qubits;
        std::vector<GateNode*> leftNodes, rightNodes;

        GateNode(unsigned nqubits) : nqubits(nqubits) {
            if (nqubits >= 3)
                throw std::runtime_error("number of qubit cannnot exceed 2");

            matrixReal.resize(1<<(2 * nqubits));
            matrixImag.resize(1<<(2 * nqubits));
            qubits.resize(nqubits);
            leftNodes = std::vector<GateNode*>(nqubits, nullptr);
            rightNodes = std::vector<GateNode*>(nqubits, nullptr);;
        }
    }; // class CircuitGraph::GateNode

    std::vector<GateNode*> leftEntry;
    std::vector<GateNode*> rightEntry;

    /// @brief Connect two nodes along qubit q.
    /// @return true if both left and right have qubit q.
    bool connectTwoNodes(GateNode* left, GateNode* right, unsigned q);

    void fuseTwoNodes(GateNode* left, GateNode* right);
public:
    CircuitGraph() : leftEntry(32, nullptr), rightEntry(32, nullptr) {}

    void addSingleQubitGate(const U3Gate& u3);

    void addTwoQubitGate(const U2qGate& u2q);

    static CircuitGraph FromQch(const qch::ast::RootNode& root);

    void transpileForCPU();

}; // class CircuitGraph

} // namespace simulation

#endif // SIMULATION_TRANSPILER_H