#include "simulation/transpiler.h"

using namespace simulation;
using namespace simulation::transpile;
using namespace qch::ast;

/// @brief C = A @ B
void matmul_complex(double* Are, double* Aim, double* Bre, double* Bim, double* Cre, double* Cim, size_t n) {
    for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
    for (size_t k = 0; k < n; k++) {
        // C_ij = A_ik B_kj
        Cre[n*i + j] = Are[n*i + k] * Bre[n*k + j]
                      -Aim[n*i + k] * Bim[n*k + j];
        Cim[n*i + j] = Are[n*i + k] * Bim[n*k + j]
                      +Aim[n*i + k] * Bre[n*k + j];
    } } }
    
}

bool CircuitGraph::connectTwoNodes(GateNode* left, GateNode* right, unsigned q) {
    bool leftFlag = (left == nullptr), rightFlag = (right == nullptr);
    if (left != nullptr) {
        for (size_t i = 0; i < left->nqubits; i++) {
            if (left->qubits[i] == q) {
                left->rightNodes[i] = right;
                leftFlag = true;
                break;
            }
        }
    }

    if (right != nullptr) {
        for (size_t i = 0; i < right->nqubits; i++) {
            if (right->qubits[i] == q) {
                right->leftNodes[i] = left;
                rightFlag = true;
                break;
            }
        }
    }

    if (leftFlag && rightFlag)
        return true;
    else {
        std::cerr << "Failed to connect two nodes!\n";
        return false;
    }
}

void CircuitGraph::addSingleQubitGate(const U3Gate& u3) {
    unsigned k = u3.k;
    if (k >= leftEntry.size()) {
        leftEntry.resize(k, nullptr);
        rightEntry.resize(k, nullptr);
    }

    // create node
    auto node = new GateNode(1);
    node->qubits[0] = k;
    for (size_t i = 0; i < 4; i++) {
        node->matrixReal[i] = u3.mat.real[i];
        node->matrixImag[i] = u3.mat.imag[i];
    }

    // update graph
    if (leftEntry[k] == nullptr) {
        leftEntry[k] = node;
    } else {
        connectTwoNodes(rightEntry[k], node, k);
    }
    rightEntry[k] = node;
}

void CircuitGraph::addTwoQubitGate(const U2qGate& u2q) {
    unsigned k = u2q.k;
    unsigned l = u2q.l;
    unsigned tmp = (k > l) ? u2q.k : u2q.l;
    if (tmp) {
        leftEntry.resize(tmp, nullptr);
        rightEntry.resize(tmp, nullptr);
    }

    // create node
    auto node = new GateNode(2);
    node->qubits[0] = k;
    node->qubits[1] = l;
    for (size_t i = 0; i < 16; i++) {
        node->matrixReal[i] = u2q.mat.real[i];
        node->matrixImag[i] = u2q.mat.imag[i];
    }
    
    // update graph 
    for (auto q : node->qubits) {
        if (leftEntry[q] == nullptr) {
            leftEntry[q] = node;
        } else {
            connectTwoNodes(rightEntry[q], node, q);
        }
        rightEntry[q] = node;
    }
}

void CircuitGraph::fuseTwoNodes(GateNode* left, GateNode *right) {
    if (left->nqubits == 1 && right->nqubits == 1) {
        unsigned q = left->qubits[0];
        auto node = new GateNode(1);
        node->qubits[0] = q;
        matmul_complex(right->matrixReal.data(), right->matrixImag.data(),
                       left->matrixReal.data(), left->matrixImag.data(),
                       node->matrixReal.data(), node->matrixImag.data(), 4);
        
        connectTwoNodes(left->leftNodes[0], node, q);
        connectTwoNodes(node, right->rightNodes[0], q);
        free(left); free(right);
    } else if (left->nqubits == 1 && right->nqubits == 2) {

    } else if (left->nqubits == 2 && right->nqubits == 1) {

    } else if (left->nqubits == 2 && right->nqubits == 2) {

    }
}

CircuitGraph CircuitGraph::FromQch(const RootNode& root) {
    CircuitGraph graph;

    auto circuit = dynamic_cast<CircuitStmt*>(root.getStmtPtr(0));
    for (size_t i = 0; i < circuit->countStmts(); i++) {
        auto gateApply = dynamic_cast<GateApplyStmt*>(circuit->getStmtPtr(i));
        if (gateApply->getName() == "u3") {
            auto u3 = U3Gate(ComplexMatrix2<>::FromEulerAngles(
                                gateApply->getParameters()[0], 
                                gateApply->getParameters()[1],
                                gateApply->getParameters()[2]),
                            gateApply->getQubits()[0]);
            graph.addSingleQubitGate(u3);
        } else if (gateApply->getName() == "u2q") {

        }
    }
    

    return graph;
}

void CircuitGraph::transpileForCPU() {
    fuseTwoNodes(leftEntry[0], rightEntry[0]);
}
