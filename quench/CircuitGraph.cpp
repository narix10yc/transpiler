#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

#include <iomanip>
#include <thread>
#include <chrono>

using namespace Color;
using namespace quench::circuit_graph;

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
    auto itRHS = std::next(itLHS);
    assert(itRHS != tile.end());
    
    auto lhs = (*itLHS)[q_];
    auto rhs = (*itRHS)[q_];

    assert(lhs);
    assert(rhs);

    std::vector<unsigned> lhsQubits;
    std::vector<unsigned> rhsQubits;
    std::vector<unsigned> blockQubits;

    auto block = new GateBlock(currentBlockId);
    currentBlockId++;

    // std::cerr << "Fuse. itLHS = " << &(*itLHS)
    //           << ", itRHS = " << &(*itRHS)
    //           << ", q = " << q_ << "\n";
    //           << "    lhs " << lhs->id << ", rhs " << rhs->id << " => block " << block->id << "\n";

    for (const auto& lData : lhs->dataVector) {
        GateNode* lhsEntry = nullptr;
        GateNode* rhsEntry = nullptr;
        const auto& q = lData.qubit;
        lhsEntry = lData.lhsEntry;
        auto it = rhs->findQubit(q);
        if (it == rhs->dataVector.end())
            rhsEntry = lData.rhsEntry;
        else
            rhsEntry = it->rhsEntry;
        block->dataVector.push_back({q, lhsEntry, rhsEntry});
        blockQubits.push_back(q);

        lhsQubits.push_back(q);
        (*itLHS)[q] = nullptr;
    }

    for (const auto& rData : rhs->dataVector) {
        const auto& q = rData.qubit;
        if (lhs->findQubit(q) == lhs->dataVector.end()) {
            block->dataVector.push_back(rData);
            blockQubits.push_back(q);
        }
        
        rhsQubits.push_back(q);
        (*itRHS)[q] = nullptr;
    }
    
    block->nqubits = block->dataVector.size();
    delete(lhs);
    delete(rhs);

    // insert block to tile
    bool vacant = true;
    for (const auto& q : rhsQubits) {
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
    for (const auto& q : lhsQubits) {
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

    row_t row{};
    for (const auto& q : blockQubits)
        row[q] = block;
    tile.insert(itRHS, row);

    return block;
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
    auto block = new GateBlock(currentBlockId, gate);
    currentBlockId++;

    // insert to tile
    auto it = tile.rbegin();
    while (it != tile.rend()) {
        bool vacant = true;
        for (const auto& q : qubits) {
            if ((*it)[q] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant)
            break;
        it++;
    }

    if (it == tile.rbegin())
        tile.push_back({nullptr});
    else
        it--;
    
    for (const auto& q : qubits)
        (*it)[q] = block;
}

size_t CircuitGraph::countGates() const {
    size_t count = 0;
    std::set<GateBlock*> s;
    for (const auto& row : tile) {
        s.clear();
        for (unsigned q = 0; q < nqubits; q++) {
            auto& block = row[q];
            if (block == nullptr || s.find(block) != s.end())
                continue;
            s.insert(block);
            count += block->countGates();
        }
    }
    return count;
}

size_t CircuitGraph::countBlocks() const {
    size_t count = 0;
    std::set<GateBlock*> s;
    for (const auto& row : tile) {
        s.clear();
        for (unsigned q = 0; q < nqubits; q++) {
            auto& block = row[q];
            if (block == nullptr || s.find(block) != s.end())
                continue;
            s.insert(block);
            count++;
        }
    }
    return count;
}

void CircuitGraph::repositionBlockUpward(tile_iter_t it_, size_t q_) {
    auto* block = (*it_)[q_];
    assert(block);

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

std::ostream& CircuitGraph::displayInfo(std::ostream& os, int verbose) const {
    os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n" << RESET;

    // os << "Number of Gates:  " << countGates() << "\n";
    os << "Number of Gates:  " << "N/A (not implemented)" << "\n";
    std::set<GateBlock*> allBlocks;
    for (const auto& row : tile) {
        for (unsigned q = 0; q < nqubits; q++) {
            if (row[q])
                allBlocks.insert(row[q]);
        }
    }

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
    std::cerr << "Dependency Analysis not implemented yet!\n";
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

void CircuitGraph::applyInOrder(std::function<void(GateBlock*)> f) const {
    std::set<GateBlock*> s;
    for (const auto& row : tile) {
        for (unsigned q = 0; q < nqubits; q++) {
            auto& block = row[q];
            if (block == nullptr || s.find(block) != s.end())
                continue;
            s.insert(block);
            f(block);
        }
    }
}