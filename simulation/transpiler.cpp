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

GateNode* CircuitGraph::addGateNode(unsigned nqubits) {
    auto node = new GateNode(nqubits);
    allNodes.insert(node);
    return node;
}

void CircuitGraph::removeGateNode(GateNode* node) {
    allNodes.erase(node);
    delete(node);
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
    auto node = addGateNode(1);
    node->qubits[0] = k;
    for (size_t i = 0; i < 4; i++)
        node->matrix.data[i] = Complex<>(u3.mat.real[i], u3.mat.imag[i]);

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
    auto node = addGateNode(2);
    node->qubits[0] = k;
    node->qubits[1] = l;
    for (size_t i = 0; i < 16; i++)
        node->matrix.data[i] = Complex<>(u2q.mat.real[i], u2q.mat.imag[i]);
    
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

CircuitGraph CircuitGraph::FromQch(const RootNode& root) {
    CircuitGraph graph;

    auto circuit = dynamic_cast<CircuitStmt*>(root.getStmtPtr(0));
    for (size_t i = 0; i < circuit->countStmts(); i++) {
        auto gateApply = dynamic_cast<GateApplyStmt*>(circuit->getStmtPtr(i));
        if (gateApply == nullptr)
            continue;
        if (gateApply->name == "u3") {
            auto u3 = U3Gate(ComplexMatrix2<>::FromEulerAngles(
                                gateApply->parameters[0], 
                                gateApply->parameters[1],
                                gateApply->parameters[2]),
                            gateApply->qubits[0]);
            graph.addSingleQubitGate(u3);
        } else if (gateApply->name == "cx") {
            auto u2q = U2qGate({{1,0,0,0, 0,0,0,1, 0,0,1,0, 0,1,0,0}, {}},
                gateApply->qubits[0], gateApply->qubits[1]);
            graph.addTwoQubitGate(u2q);
        }
    }

    return graph;
}

unsigned CircuitGraph::absorbNeighbouringSingleQubitGates(GateNode* node) {
    unsigned nFused = 0;
    if (node->nqubits == 1) {
        GateNode* left = node->leftNodes[0];
        if (left != nullptr && left->nqubits == 1) {
            node->matrix = node->matrix.matmul(left->matrix);
            connectTwoNodes(left->leftNodes[0], node, node->qubits[0]);
            removeGateNode(left);
            nFused += 1;
        }
        GateNode* right = node->rightNodes[0];
        if (right != nullptr && right->nqubits == 1) {
            node->matrix = right->matrix.matmul(node->matrix);
            connectTwoNodes(node, right->rightNodes[0], node->qubits[0]);
            removeGateNode(right);
            nFused += 1;
        }
    } else if (node->nqubits == 2) {
        GateNode* left0 = node->leftNodes[0];
        if (left0 != nullptr && left0->nqubits == 1) {
            // R @ (I otimes L)
            node->matrix = node->matrix.matmul(left0->matrix.leftKronI());
            connectTwoNodes(left0->leftNodes[0], node, node->qubits[0]);
            removeGateNode(left0);
            nFused += 1;
        }
        GateNode* left1 = node->leftNodes[1];
        if (left1 != nullptr && left1->nqubits == 1) {
            // R @ (L otimes I)
            node->matrix = node->matrix.matmul(left1->matrix.rightKronI());
            connectTwoNodes(left1->leftNodes[0], node, node->qubits[1]);
            removeGateNode(left1);
            nFused += 1;
        }

        GateNode* right0 = node->rightNodes[0];
        if (right0 != nullptr && right0->nqubits == 1) {
            // (I otimes R) @ L
            node->matrix = right0->matrix.leftKronI().matmul(node->matrix);
            connectTwoNodes(node, right0->rightNodes[0], node->qubits[0]);
            removeGateNode(right0);
            nFused += 1;
        }
        GateNode* right1 = node->rightNodes[1];
        if (right1 != nullptr && right1->nqubits == 1) {
            // (R otimes I) @ L
            node->matrix = right1->matrix.rightKronI().matmul(node->matrix);
            connectTwoNodes(node, right1->rightNodes[0], node->qubits[1]);
            removeGateNode(right1);
            nFused += 1;
        }
    }
    
    return nFused;
}

 
void CircuitGraph::transpileForCPU() {
    // step 1: absorb single-qubit gates
    bool updates = false;
    do {
        updates = false;
        for (GateNode* node : allNodes) {
            auto nFused = absorbNeighbouringSingleQubitGates(node);
            if (nFused > 0) {
                std::cerr << nFused << " gates fused\n";
                updates = true;
                break;
            }
        }
    } while (updates);

    std::cerr << "fusion step 1 finished! " << allNodes.size() << " nodes remaining\n";


    // step 2: fuse two-qubit gates
}

namespace {
    double approximate(double x, double thres=1e-8) {
        if (abs(x) < thres)
            return 0;
        if (abs(x - 1) < thres)
            return 1;
        if (abs(x + 1) < thres)
            return -1;
        return x;
    }
}

RootNode CircuitGraph::toQch() const {
    RootNode root;
    auto circuit = std::make_unique<CircuitStmt>("transpiled");

    for (GateNode* node : allNodes) {
        std::string name = (node->nqubits == 1) ? "u3" : "u2q";
        auto gateApply = std::make_unique<GateApplyStmt>(name);
        for (auto p : node->matrix.data) {
            gateApply->addParameter(approximate(p.real));
            gateApply->addParameter(approximate(p.imag));
        }
        for (auto q : node->qubits) {
            gateApply->addTargetQubit(q);
        }

        circuit->addStmt(std::move(gateApply));
    }
    root.addStmt(std::move(circuit));
    return root;
}