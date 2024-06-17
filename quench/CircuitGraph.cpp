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
            rhsIt->rhsBlock = data.rhsBlock;
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
            rhsBlock->connect(block);

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

std::ostream& CircuitGraph::print(std::ostream& os) const {
    if (allBlocks.empty())
        return os;
    std::cerr << "printing...\n";

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
    print(std::cerr);
    bool hasChange = false;
    GateBlock* rhsBlock;
    while (true) {
        hasChange = false;
        std::cerr << "Big loop!\n";
        for (auto* block : allBlocks) {
            assert(block != nullptr);
            std::cerr << "checking block " << block->id << "\n";
            if (block->nqubits == 1) {
                rhsBlock = block->dataVector[0].rhsBlock;
                if (rhsBlock == nullptr)
                    continue;
                std::cerr << "fusing block " << block->id << " with " << rhsBlock->id << "\n";
                block->fuseWithRHS(rhsBlock);
                std::cerr << "done fusing " << block->id << " with " << rhsBlock->id << "\n";

                destroyBlock(block);

                // delete(lhsBlock);
                hasChange = true;
                break;
            }
            // } else {
            //     // nqubits == 2
            //     lhsBlock = block->dataVector[0].lhsBlock;
            //     if (lhsBlock->nqubits == 1) {
            //         lhsBlock->fuseWithRHS(block);
            //         destroyBlock(lhsBlock);
            //         hasChange = true;
            //     } else {}
            // }
        }

        if (!hasChange)
            break;
    }
}

void CircuitGraph::greedyGateFusion() {
    std::cerr << "Greedy Gate Fusion not implemented yet!\n";
}