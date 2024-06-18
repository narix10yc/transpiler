#include "quench/CircuitGraph.h"
#include <iomanip>

using namespace quench::circuit_graph;

void GateBlock::fuseWithRHS(GateBlock* rhsBlock) {
    assert(rhsBlock != nullptr);

    for (const auto& data : dataVector) {
        auto rhsIt = rhsBlock->findQubit(data.qubit);
        if (rhsIt == rhsBlock->dataVector.end()) {
            rhsBlock->dataVector.push_back(data);
        } else {
            rhsIt->lhsBlock = data.lhsBlock;
        }
    }
}

void GateBlock::fuseWithLHS(GateBlock* lhsBlock) {
    assert(lhsBlock != nullptr);

    for (const auto& data : dataVector) {
        auto lhsIt = lhsBlock->findQubit(data.qubit);
        if (lhsIt == lhsBlock->dataVector.end()) {
            lhsBlock->dataVector.push_back(data);
        } else {
            lhsIt->rhsBlock = data.rhsBlock;
        }
    }
}

void CircuitGraph::addGate(const cas::GateMatrix& matrix,
                           const std::vector<unsigned>& qubits)
{
    assert(matrix.nqubits == qubits.size());

    // update nqubits
    for (const auto& q : qubits) {
        if (q >= nqubits)
            nqubits = q + 1;
    }

    // create gate and block
    auto gate = new GateNode(matrix, qubits);
    auto block = createBlock(gate);

    // connect
    for (const auto& q : qubits) {
        if (lhsEntry[q] == nullptr) {
            assert(rhsEntry[q] == nullptr);
            lhsEntry[q] = block;
            rhsEntry[q] = block;
        } else {
            auto rhsBlock = rhsEntry[q];
            auto it = rhsBlock->findQubit(q);
            assert(it != rhsBlock->dataVector.end());

            it->rhsEntry->connect(gate, q);
            rhsBlock->connect(block, q);

            rhsEntry[q] = block;
        }
    }
}

GateBlock* CircuitGraph::createBlock(GateNode* gate) {
    auto block = new GateBlock(currentBlockId, gate);
    currentBlockId++;
    allBlocks.insert(block);
    return block;
}

void CircuitGraph::destroyBlock(GateBlock* block) {
    assert(block != nullptr);
    assert(allBlocks.find(block) != allBlocks.end());

    for (auto& data : block->dataVector) {
        if (data.lhsBlock && data.rhsBlock) {
            data.lhsBlock->connect(data.rhsBlock, data.qubit);
        } else if (data.lhsBlock) {
            auto it = data.lhsBlock->findQubit(data.qubit);
            (*it).rhsBlock = nullptr;
        } else if (data.rhsBlock) {
            auto it = data.rhsBlock->findQubit(data.qubit);
            (*it).lhsBlock = nullptr;
        }
    }

    allBlocks.erase(block);
    delete(block);
}

// TODO: More efficient way: store 'lastAvailableRow' array
std::ostream& CircuitGraph::print(std::ostream& os) const {
    if (allBlocks.empty())
        return os;

    std::vector<std::vector<int>> tile;
    auto appendOneLine = [&, q=nqubits]() {
        tile.push_back(std::vector<int>(static_cast<size_t>(q), -1));
    };

    bool vancant = true;
    size_t lineIdx;
    for (const auto* block : allBlocks) {
        // find which line to place the block
        if (tile.empty())
        {
            appendOneLine();
            lineIdx = 0;
        } 
        else
        {
            lineIdx = tile.size() - 1;
            while (true) {
                vancant = true;
                for (const auto& data : block->dataVector) {
                    if (tile[lineIdx][data.qubit] >= 0) {
                        vancant = false;
                        break;
                    }
                }
                if (vancant && lineIdx > 0) {
                    lineIdx--;
                    continue;
                }
                if (!vancant) {
                    if (lineIdx == tile.size() - 1)
                        appendOneLine();
                    lineIdx++;
                } else { // if (vacent && lineIdx == 0)
                    lineIdx = 0;
                }
                break;
            }
        }
        for (const auto& data : block->dataVector)
            tile[lineIdx][data.qubit] = block->id;
    }

    int width = static_cast<int>(std::log10(currentBlockId)) + 1;
    if ((width & 1) == 0)
        width++;

    std::string vbar = std::string(width/2, ' ') + "|" + std::string(width/2+1, ' ');

    for (const auto& line : tile) {
        for (unsigned i = 0; i < nqubits; i++) {
            if (line[i] < 0)
                os << vbar;
            else
                os << std::setfill('0') << std::setw(width)
                   << line[i] << " ";
        }
        os << "\n";
    }
    return os;
}

void CircuitGraph::dependencyAnalysis() {
    std::cerr << "Dependency Analysis not implemented yet!\n";
}

void CircuitGraph::fuseToTwoQubitGates() {
    bool hasChange = false;
    GateBlock* lhsBlock;
    GateBlock* rhsBlock;
    while (true) {
        hasChange = false;
        print(std::cerr) << "\n";
        for (auto* block : allBlocks) {
            if (block->nqubits == 1)
                continue;
            // nqubits == 2
            if ((lhsBlock = block->dataVector[0].lhsBlock)
                 && lhsBlock->nqubits == 1) {
                lhsBlock->fuseWithRHS(block);
                destroyBlock(lhsBlock);
                hasChange = true;
            }
            if ((lhsBlock = block->dataVector[1].lhsBlock)
                 && lhsBlock->nqubits == 1) {
                lhsBlock->fuseWithRHS(block);
                destroyBlock(lhsBlock);
                hasChange = true;
            }
            if ((rhsBlock = block->dataVector[0].rhsBlock)
                 && rhsBlock->nqubits == 1) {
                rhsBlock->fuseWithLHS(block);
                destroyBlock(rhsBlock);
                hasChange = true;
            }
            if ((rhsBlock = block->dataVector[1].rhsBlock)
                 && rhsBlock->nqubits == 1) {
                rhsBlock->fuseWithLHS(block);
                destroyBlock(rhsBlock);
                hasChange = true;
            }
        }

        if (!hasChange)
            break;
    }
}

void CircuitGraph::greedyGateFusion() {
    std::cerr << "Greedy Gate Fusion not implemented yet!\n";
}