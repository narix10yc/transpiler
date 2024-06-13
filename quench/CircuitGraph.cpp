#include "quench/CircuitGraph.h"
#include <iomanip>

using namespace quench::circuit_graph;

void CircuitGraph::addGate(const cas::GateMatrix& matrix,
                           const std::vector<unsigned>& qubits)
{
    assert(matrix.nqubits == qubits.size());

    // update nqubits
    for (const auto& q : qubits) {
        if (q >= nqubits)
            nqubits = q + 1;
    }

    // create gate
    auto gate = new GateNode(matrix);
    for (const auto& q : qubits) {
        if (lhsEntry[q] == nullptr) {
            gate->dataVector.push_back({q, nullptr, nullptr});
            lhsEntry[q] = gate;
            rhsEntry[q] = gate;
        } else {
            auto rhsGate = rhsEntry[q];
            rhsGate->updateRHS(gate, q);
            gate->dataVector.push_back({q, rhsGate, nullptr});
            rhsEntry[q] = gate;
        }
    }

    // create block
    auto block = new GateBlock(currentBlockId, gate);
    currentBlockId++;

    if (tile.empty()) {
        tile.push_back({});
        auto& back = tile.back();
        for (const auto& q : qubits)
            back[q] = block;
        return;
    }

    auto& back = tile.back();
    for (const auto& q : qubits) {
        if (back[q] != nullptr) {
            tile.push_back({});
            break;
        }
    }

    for (const auto& q : qubits)
        tile.back()[q] = block;
    
}

std::ostream& CircuitGraph::print(std::ostream& os) const {
    if (tile.empty())
        return os;

    int width = static_cast<int>(std::log10(currentBlockId)) + 1;
    if ((width & 1) == 0)
        width++;

    std::string vbar = std::string(width/2, ' ') + "|" + std::string(width/2+1, ' ');

    for (const auto& line : tile) {
        for (unsigned i = 0; i < nqubits; i++) {
            if (line[i] == nullptr)
                os << vbar;
            else
                os << std::setfill('0') << std::setw(width)
                   << line[i]->id << " ";
        }
        os << "\n";
    }
}

void CircuitGraph::dependencyAnalysis() {
    std::cerr << "Dependency Analysis not implemented yet!\n";
}

void CircuitGraph::fuseToTwoQubitGates() {

}

void CircuitGraph::greedyGateFusion() {
    std::cerr << "Greedy Gate Fusion not implemented yet!\n";
}