#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

#include <iomanip>
#include <thread>
#include <chrono>
#include <map>
#include <deque>

using namespace Color;
using namespace quench::circuit_graph;

int GateNode::connect(GateNode* rhsGate, int q) {
    assert(rhsGate);

    if (q >= 0) {
        auto myIt = findQubit(static_cast<unsigned>(q));
        assert(myIt != dataVector.end());
            // return 0;
        auto rhsIt = rhsGate->findQubit(static_cast<unsigned>(q));
        assert(rhsIt != rhsGate->dataVector.end());
            // return 0;
        
        myIt->rhsGate = rhsGate;
        rhsIt->lhsGate = this;
        return 1;
    }
    int count = 0;
    for (auto& data : dataVector) {
        auto rhsIt = rhsGate->findQubit(data.qubit);
        if (rhsIt == rhsGate->dataVector.end())
            continue;
        data.rhsGate = rhsGate;
        rhsIt->lhsGate = this;
        count++;
    }
    return count;
}

int GateBlock::connect(GateBlock* rhsBlock, int q) {
    assert(rhsBlock);
    if (q >= 0) {
        auto myIt = findQubit(static_cast<unsigned>(q));
        assert(myIt != dataVector.end());
            // return 0;
        auto rhsIt = rhsBlock->findQubit(static_cast<unsigned>(q));
        assert(rhsIt != rhsBlock->dataVector.end());
            // return 0;
        myIt->rhsEntry->connect(rhsIt->lhsEntry, q);
        return 1;
    }
    int count = 0;
    for (auto& data : dataVector) {
        auto rhsIt = rhsBlock->findQubit(data.qubit);
        if (rhsIt == rhsBlock->dataVector.end())
            continue;
        data.rhsEntry->connect(rhsIt->lhsEntry, data.qubit);
        count++;
    }
    return count;
}

int CircuitGraph::checkFuseCondition(tile_const_iter_t itLHS, size_t q_) const {
    assert((*itLHS)[q_]);

    auto itRHS = std::next(itLHS);
    if (itRHS == tile.cend())
        return -1000;
    
    GateBlock* lhs = (*itLHS)[q_];
    GateBlock* rhs = (*itRHS)[q_];
    if (!(lhs && rhs))
        return -100;

    std::set<unsigned> qubits;
    for (const auto& data : lhs->dataVector)
        qubits.insert(data.qubit);
    for (const auto& data : rhs->dataVector)
        qubits.insert(data.qubit);
    
    return qubits.size();
}

GateBlock* CircuitGraph::fuse(tile_iter_t itLHS, size_t q_) {
    const auto itRHS = std::next(itLHS);
    assert(itRHS != tile.end());
    
    auto lhs = (*itLHS)[q_];
    auto rhs = (*itRHS)[q_];

    assert(lhs);
    assert(rhs);

    std::vector<unsigned> blockQubits;

    auto block = new GateBlock(currentBlockId);
    currentBlockId++;

    // std::cerr << "Fuse. itLHS = " << &(*itLHS)
    //           << ", itRHS = " << &(*itRHS)
    //           << ", q = " << q_ << "\n"
    //           << "    lhs " << lhs->id << ", rhs " << rhs->id << " => block " << block->id << "\n";
    

    for (const auto& lData : lhs->dataVector) {
        const auto& q = lData.qubit;
        (*itLHS)[q] = nullptr;

        GateNode* lhsEntry = lData.lhsEntry;
        GateNode* rhsEntry;
        auto it = rhs->findQubit(q);
        if (it == rhs->dataVector.end())
            rhsEntry = lData.rhsEntry;
        else
            rhsEntry = it->rhsEntry;

        assert(lhsEntry);
        assert(rhsEntry);

        block->dataVector.push_back({q, lhsEntry, rhsEntry});

        blockQubits.push_back(q);
    }

    for (const auto& rData : rhs->dataVector) {
        const auto& q = rData.qubit;
        (*itRHS)[q] = nullptr;

        if (lhs->findQubit(q) == lhs->dataVector.end()) {
            block->dataVector.push_back(rData);
            blockQubits.push_back(q);
        }
    }
    
    block->nqubits = block->dataVector.size();
    delete(lhs);
    delete(rhs);
    
    // std::cerr << BLUE_FG;
    // block->displayInfo(std::cerr) << RESET;

    // insert block to tile
    bool vacant = true;
    for (const auto& q : blockQubits) {
        if ((*itLHS)[q] != nullptr) {
            vacant = false;
            break;
        }
    }
    if (vacant) {
        // insert at itLHS
        for (const auto& q : blockQubits) {
            assert((*itLHS)[q] == nullptr);
            (*itLHS)[q] = block;
        }
        return block;
    }

    vacant = true;
    for (const auto& q : blockQubits) {
        if ((*itRHS)[q] != nullptr) {
            vacant = false;
            break;
        }
    }
    if (vacant) {
        // insert at itRHS
        for (const auto& q : blockQubits) {
            assert((*itRHS)[q] == nullptr);
            (*itRHS)[q] = block;
        }
        return block;
    }

    row_t row{nullptr};
    for (const auto& q : blockQubits)
        row[q] = block;
    tile.insert(itRHS, row);

    return block;
}

void CircuitGraph::addGate(const quantum_gate::GateMatrix& matrix,
                           const std::vector<unsigned>& qubits)
{
    assert(matrix.nqubits == qubits.size());

    // update nqubits
    for (const auto& q : qubits) {
        if (q >= nqubits)
            nqubits = q + 1;
    }

    // create gate and block
    auto gate = new GateNode(currentBlockId, matrix, qubits);
    auto block = new GateBlock(currentBlockId, gate);
    currentBlockId++;

    // insert block to tile
    auto rit = tile.rbegin();
    while (rit != tile.rend()) {
        bool vacant = true;
        for (const auto& q : qubits) {
            if ((*rit)[q] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant)
            break;
        rit++;
    }
    if (rit == tile.rbegin())
        tile.push_back({nullptr});
    else
        rit--;
    for (const auto& q : qubits)
        (*rit)[q] = block;

    // set up gate and block connections
    auto itRHS = --rit.base();
    if (itRHS == tile.begin())
        return;
    
    tile_iter_t itLHS;
    GateBlock* lhsBlock;
    for (const auto& q : qubits) {
        itLHS = itRHS;
        while (--itLHS != tile.begin()) {
            if ((*itLHS)[q])
                break;
        }
        if ((lhsBlock = (*itLHS)[q]) == nullptr)
            continue;
        lhsBlock->connect(block, q);
    }

}

std::vector<GateBlock*> CircuitGraph::getAllBlocks() const {
    std::vector<GateBlock*> allBlocks;
    std::vector<GateBlock*> rowBlocks;
    for (const auto& row : tile) {
        rowBlocks.clear();
        for (const auto& block : row) {
            if (block == nullptr)
                continue;
            if (std::find(rowBlocks.begin(), rowBlocks.end(), block) == rowBlocks.end())
                rowBlocks.push_back(block);
        }
        for (const auto& block : rowBlocks)
            allBlocks.push_back(block);
    }
    return allBlocks;
}

void CircuitGraph::repositionBlockUpward(tile_iter_t it_, size_t q_) {
    auto* block = (*it_)[q_];
    assert(block);

    if (it_ == tile.begin())
        return;

    bool vacant = true;
    auto it = it_;
    do {
        it--;
        vacant = true;
        for (const auto& data : block->dataVector) {
            if ((*it)[data.qubit] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant) {
            it++;
            break;
        }
    } while (it != tile.begin());

    if (it == it_)
        return;
    for (const auto& data : block->dataVector) {
        const auto& q = data.qubit;
        (*it_)[q] = nullptr;
        (*it)[q] = block;
    }

    return;
}

void CircuitGraph::repositionBlockDownward(tile_riter_t it_, size_t q_) {
    auto* block = (*it_)[q_];
    assert(block);

    if (it_ == tile.rbegin())
        return;

    bool vacant = true;
    auto it = it_;
    do {
        it--;
        vacant = true;
        for (const auto& data : block->dataVector) {
            if ((*it)[data.qubit] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant) {
            it++;
            break;
        }
    } while (it != tile.rbegin());

    if (it == it_)
        return;
    for (const auto& data : block->dataVector) {
        const auto& q = data.qubit;
        (*it_)[q] = nullptr;
        (*it)[q] = block;
    }
    return;
}

void CircuitGraph::eraseEmptyRows() {
    auto it = tile.cbegin();
    bool empty;
    while (it != tile.cend()) {
        empty = true;
        for (unsigned q = 0; q < nqubits; q++) {
            if ((*it)[q]) {
                empty = false;
                break;
            }
        }
        if (empty)
            it = tile.erase(it);
        else
            it++;
    }
}

void CircuitGraph::updateTileUpward() {
    eraseEmptyRows();
    auto it = tile.begin();
    while (it != tile.end()) {
        for (unsigned q = 0; q < nqubits; q++) {
            if ((*it)[q])
                repositionBlockUpward(it, q);
        }
        it++;
    }
    eraseEmptyRows();
}

void CircuitGraph::updateTileDownward() {
    eraseEmptyRows();

    auto it = tile.rbegin();
    while (it != tile.rend()) {
        for (unsigned q = 0; q < nqubits; q++) {
            if ((*it)[q])
                repositionBlockDownward(it, q);
        }
        it++;
    }
    eraseEmptyRows();
}

std::ostream& CircuitGraph::print(std::ostream& os, int verbose) const {
    int width = static_cast<int>(std::log10(currentBlockId)) + 1;
    if ((width & 1) == 0)
        width++;

    std::string vbar = std::string(width/2, ' ') + "|" + std::string(width/2+1, ' ');

    auto it = tile.cbegin();
    while (it != tile.cend()) {
        if (verbose > 1)
            os << &(*it) << ": ";
        for (unsigned q = 0; q < nqubits; q++) {
            auto* block = (*it)[q];
            if (block == nullptr)
                os << vbar;
            else
                os << std::setw(width) << std::setfill('0') << block->id << " ";
        }
        os << "\n";
        it++;
    }
    return os;
}

std::ostream& GateBlock::displayInfo(std::ostream& os) const {
    os << "Block " << id << ": [";
    for (const auto& data : dataVector) {
        os << "(" << data.qubit << ":";
        GateNode* gate = data.lhsEntry;
        assert(gate);
        os << gate << ",";
        while (gate != data.rhsEntry) {
            gate = gate->findRHS(data.qubit);
            assert(gate);
            os << gate << ",";
        }
        os << "),";
    }
    return os << "]\n";
}

std::ostream& CircuitGraph::displayInfo(std::ostream& os, int verbose) const {
    os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n" << RESET;

    os << "Number of Gates:  " << countGates() << "\n";
    const auto allBlocks = getAllBlocks();
    os << "Number of Blocks: " << allBlocks.size() << "\n";
    os << "Circuit Depth:    " << tile.size() << "\n";

    if (verbose > 1) {
        os << "Block Statistics:\n";
        std::map<unsigned, unsigned> map;
        for (const auto& block : allBlocks) {
            const auto& nqubits = block->nqubits;
            if (map.find(nqubits) == map.end())
                map[nqubits] = 1;
            else
                map[nqubits] ++;
        }
        for (const auto& pair : map)
            os << "  " << pair.first << "-qubit: " << pair.second << "\n";
    }

    os << CYAN_FG << "=====================================\n" << RESET;
    return os;
}

void CircuitGraph::dependencyAnalysis() {
    assert(false && "Not implemented yet!");
}

void CircuitGraph::fuseToTwoQubitGates() {
    if (tile.size() < 2)
        return;

    GateBlock* lhsBlock;
    GateBlock* rhsBlock;

    bool hasChange = true;
    tile_iter_t itLHS, itRHS;
    while (true) {
        hasChange = false;
        itLHS = tile.begin();
        itRHS = std::next(itLHS);
        while (itRHS != tile.end()) {
            for (unsigned q = 0; q < nqubits; q++) {
                if (!((lhsBlock = (*itLHS)[q]) && (rhsBlock = (*itRHS)[q])))
                    continue;
                if (lhsBlock->nqubits == 2 && rhsBlock->nqubits == 2 && !lhsBlock->hasSameTargets(*rhsBlock))
                    continue;
                fuse(itLHS, q);
                hasChange = true;
                print(std::cerr, 2) << "\n\n";
            }
            itRHS = std::next(++itLHS);
        }
        if (hasChange) {
            updateTileUpward();
            std::cerr << "update tile\n";
            print(std::cerr, 2) << "\n\n";
        } else { break; }
    }
}

void CircuitGraph::greedyGateFusion(int maxNQubits) {
    if (tile.size() < 2)
        return;

    GateBlock* lhsBlock;
    GateBlock* rhsBlock;

    bool hasChange = true;
    tile_iter_t it;
    while (true) {
        hasChange = false;
        it = tile.begin();
        while (std::next(it) != tile.end()) {
            for (unsigned q = 0; q < nqubits; q++) {
                if ((*it)[q] == nullptr)
                    continue;
                int check = checkFuseCondition(it, q);
                if (check <= 0) {
                    repositionBlockDownward(it, q);
                    continue;
                }
                if (check > maxNQubits)
                    continue;
                fuse(it, q);
                hasChange = true;
                // print(std::cerr, 2) << "\n\n";
            }
            it++;
        }
        if (hasChange) {
            updateTileUpward();
            // std::cerr << "update tile\n";
            // print(std::cerr, 2) << "\n\n";
        } else { break; }
    }
}

std::vector<GateNode*> GateBlock::getOrderedGates() const {
    std::deque<GateNode*> queue;
    // vector should be more efficient as we expect small size here
    std::vector<GateNode*> gates;
    for (const auto& data : dataVector) {
        if (std::find(queue.begin(), queue.end(), data.rhsEntry) == queue.end())
            queue.push_back(data.rhsEntry);
    }
    
    while (!queue.empty()) {
        const auto& gate = queue.back();
        if (std::find(gates.begin(), gates.end(), gate) != gates.end()) {
            queue.pop_back();
            continue;
        }
        std::vector<GateNode*> higherPriorityGates;
        for (const auto& data : this->dataVector) {
            if (gate == data.lhsEntry)
                continue;
            auto it = gate->findQubit(data.qubit);
            if (it == gate->dataVector.end())
                continue;
            assert(it->lhsGate);
            if (std::find(gates.begin(), gates.end(), it->lhsGate) == gates.end())
                higherPriorityGates.push_back(it->lhsGate);
        }

        if (higherPriorityGates.empty()) {
            queue.pop_back();
            gates.push_back(gate);
        } else {
            for (const auto& g : higherPriorityGates)
                queue.push_back(g);    
        }
    }
    return gates;
}

using QuantumGate = quench::quantum_gate::QuantumGate;

QuantumGate GateNode::toQuantumGate() const {
    return QuantumGate(gateMatrix, getQubits());
}

QuantumGate GateBlock::toQuantumGate() const {
    const auto gateNodes = getOrderedGates();
    assert(!gateNodes.empty());

    auto gate = gateNodes[0]->toQuantumGate();
    for (unsigned i = 1; i < gateNodes.size(); i++)
        gate = gate.lmatmul(gateNodes[i]->toQuantumGate());
    
    return gate;
}
