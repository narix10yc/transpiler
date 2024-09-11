#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <iomanip>
#include <thread>
#include <chrono>
#include <map>
#include <deque>
#include <numeric>

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

CircuitGraph::tile_iter_t
CircuitGraph::insertBlock(tile_iter_t it, GateBlock* block) {
    assert(block);

    const auto qubits = block->getQubits();
    bool vacant = true;
    for (const auto& q : qubits) {
        if ((*it)[q] != nullptr) {
            vacant = false;
            break;
        }
    }
    if (vacant) {
        // insert at it
        for (const auto& q : qubits) {
            assert((*it)[q] == nullptr);
            (*it)[q] = block;
        }
        return it;
    }

    it++;
    if (it == tile.end()) {
        row_t row {nullptr};
        for (const auto& q : qubits)
            row[q] = block;
        return tile.insert(it, row);
    }

    vacant = true;
    for (const auto& q : qubits) {
        if ((*it)[q] != nullptr) {
            vacant = false;
            break;
        }
    }
    if (vacant) {
        // insert at it
        for (const auto& q : qubits) {
            assert((*it)[q] == nullptr);
            (*it)[q] = block;
        }
        return it;
    }

    row_t row {nullptr};
    for (const auto& q : qubits)
        row[q] = block;
    return tile.insert(it, row);
}

GateBlock* CircuitGraph::fusionCandidate(GateBlock* lhs, GateBlock* rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;
    
    if (lhs == rhs)
        return nullptr;
    
    assert(lhs->quantumGate != nullptr);
    assert(rhs->quantumGate != nullptr);

    // candidate block
    auto block = new GateBlock(currentBlockId);
    currentBlockId++;

    // std::cerr << "Trying to fuse "
            //   << "lhs " << lhs->id << " and rhs " << rhs->id
            //   << " => candidate block " << block->id << "\n";

    std::vector<unsigned> blockQubits;
    for (const auto& lData : lhs->dataVector) {
        const auto& q = lData.qubit;

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
        if (lhs->findQubit(q) == lhs->dataVector.end()) {
            block->dataVector.push_back(rData);
            blockQubits.push_back(q);
        }
    }
    
    block->nqubits = block->dataVector.size();

    // check fusion condition
    if (block->nqubits > fusionConfig.maxNQubits) {
        // std::cerr << CYAN_FG << "Rejected due to maxNQubits\n" << RESET;
        delete(block);
        return nullptr;
    }
    
    auto blockQuantumGate = rhs->quantumGate->lmatmul(*(lhs->quantumGate));
    if (fusionConfig.maxOpCount >= 0 && 
            blockQuantumGate.opCount(fusionConfig.zeroSkippingThreshold) > fusionConfig.maxOpCount) {
        // std::cerr << CYAN_FG << "Rejected due to OpCount\n" << RESET;
        delete(block);
        return nullptr;
    }

    // accept candidate
    // std::cerr << GREEN_FG << "Fusion accepted!\n" << RESET;
    block->quantumGate = std::make_unique<QuantumGate>(blockQuantumGate);
    return block;
}

GateBlock* CircuitGraph::tryFuseConnectedConsecutive(tile_iter_t itLHS, size_t q_) {
    const auto itRHS = std::next(itLHS);
    assert(itRHS != tile.end());
    
    GateBlock* lhs = (*itLHS)[q_];
    GateBlock* rhs = (*itRHS)[q_];

    if (!lhs || !rhs)
        return nullptr;
  
    GateBlock* block = fusionCandidate(lhs, rhs);
    if (block == nullptr)
        return nullptr;

    // std::cerr << BLUE_FG;
    // lhs->displayInfo(std::cerr);
    // rhs->displayInfo(std::cerr);
    // block->displayInfo(std::cerr) << RESET;

    for (const auto& lData : lhs->dataVector)
        (*itLHS)[lData.qubit] = nullptr;
    for (const auto& rData : rhs->dataVector)
        (*itRHS)[rData.qubit] = nullptr;

    delete(lhs);
    delete(rhs);

    // insert block to tile
    auto itBlock = insertBlock(itLHS, block);
    for (const auto q : block->getQubits()) {
        assert((*itBlock)[q] == block);
    }
    // updateTileUpward();

    return block;
}

GateBlock* CircuitGraph::tryFuseSameRow(tile_iter_t tileIt, size_t q0) {
    auto block0 = (*tileIt)[q0];
    if (block0 == nullptr)
        return nullptr;
    
    for (unsigned q1 = 0; q1 < nqubits; q1++) {
        auto* block1 = (*tileIt)[q1];
        auto* fusedBlock = fusionCandidate(block0, block1);
        if (fusedBlock == nullptr)
            continue;
        for (const auto q : fusedBlock->getQubits()) {
            (*tileIt)[q] = fusedBlock;
        }
        delete(block0);
        delete(block1);
        return fusedBlock;
    }
    return nullptr;

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

CircuitGraph::tile_iter_t
CircuitGraph::repositionBlockUpward(tile_iter_t it, size_t q_) {
    auto* block = (*it)[q_];
    assert(block);

    if (it == tile.begin())
        return it;

    bool vacant = true;
    auto newIt = it;
    while (true) {
        newIt--;
        vacant = true;
        for (const auto& data : block->dataVector) {
            if ((*newIt)[data.qubit] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant) {
            newIt++;
            break;
        }
        if (newIt == tile.begin())
            break;
    }

    if (newIt == it)
        return newIt;
    for (const auto& data : block->dataVector) {
        const auto& q = data.qubit;
        (*it)[q] = nullptr;
        (*newIt)[q] = block;
    }
    return newIt;
}

CircuitGraph::tile_riter_t
CircuitGraph::repositionBlockDownward(tile_riter_t it, size_t q_) {
    auto* block = (*it)[q_];
    assert(block);

    if (it == tile.rbegin())
        return it;

    bool vacant = true;
    auto newIt = it;
    while (true) {
        newIt--;
        vacant = true;
        for (const auto& data : block->dataVector) {
            if ((*newIt)[data.qubit] != nullptr) {
                vacant = false;
                break;
            }
        }
        if (!vacant) {
            newIt++;
            break;
        }
        if (newIt == tile.rbegin())
            break;
    }

    if (newIt == it)
        return newIt;
    for (const auto& data : block->dataVector) {
        const auto& q = data.qubit;
        (*it)[q] = nullptr;
        (*newIt)[q] = block;
    }
    return newIt;
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
        os << gate->id << ",";
        while (gate != data.rhsEntry) {
            gate = gate->findRHS(data.qubit);
            assert(gate);
            os << gate->id << ",";
        }
        os << "),";
    }
    return os << "]\n";
}

std::vector<int> CircuitGraph::getBlockSizes() const {
    std::vector<int> sizes(nqubits + 1, 0);
    const auto allBlocks = getAllBlocks();
    int largestSize = 0;
    for (const auto* b : allBlocks) {
        sizes[b->nqubits]++;
        if (b->nqubits > largestSize)
            largestSize = b->nqubits;
    }
    sizes.resize(largestSize+1);
    return sizes;
}

std::vector<std::vector<int>> CircuitGraph::getBlockOpCountHistogram() const {
    const auto allBlocks = getAllBlocks();
    int largestSize = 0;
    for (const auto* b : allBlocks) {
        if (b->nqubits > largestSize)
            largestSize = b->nqubits;
    }
    std::vector<std::vector<int>> hist(largestSize+1);
    for (unsigned q = 1; q < largestSize+1; q++)
        hist[q].resize(q, 0);
    
    for (const auto* b : allBlocks) {
        const int q = b->nqubits;
        int catagory = 0;
        int opCount = b->quantumGate->opCount();
        while ((1 << (2 * catagory + 3)) < opCount)
            catagory++;
        
        hist[q][catagory]++;
    }
    return hist;   
}

std::ostream& CircuitGraph::displayInfo(std::ostream& os, int verbose) const {
    os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n" << RESET;

    os << "Number of Gates:  " << countGates() << "\n";
    const auto allBlocks = getAllBlocks();
    auto nBlocks = allBlocks.size();
    os << "Number of Blocks: " << nBlocks << "\n";
    auto totalOp = countTotalOps();
    os << "Total Op Count:   " << totalOp << "\n";
    os << "Avergae Op Count: " << std::fixed << std::setprecision(1)
       << static_cast<double>(totalOp) / nBlocks << "\n";
    os << "Circuit Depth:    " << tile.size() << "\n";

    if (verbose > 3) {
        os << "Block Sizes Count:\n";
        std::vector<std::vector<int>> vec(nqubits + 1);
        const auto allBlocks = getAllBlocks();
        for (const auto* block : allBlocks)
            vec[block->nqubits].push_back(block->id);
        
        for (unsigned q = 1; q < vec.size(); q++) {
            if (vec[q].empty())
                continue;
            os << "  " << q << "-qubit: count " << vec[q].size() << " ";
            utils::printVector(vec[q], os) << "\n";
        }
    }
    else if (verbose > 2) {
        os << "Block Statistics:\n";
        const auto hist = getBlockOpCountHistogram();
        for (unsigned q = 1; q < hist.size(); q++) {
            auto count = std::reduce(hist[q].begin(), hist[q].end());
            if (count == 0)
                continue;
            os << "  " << q << "-qubit count = " << count << "; hist: ";
            utils::printVector(hist[q], os) << "\n";
        }
    }
    else if (verbose > 1) {
        os << "Block Sizes Count:\n";
        const auto sizes = getBlockSizes();
        for (unsigned q = 1; q < sizes.size(); q++) {
            if (sizes[q] <= 0)
                continue;
            os << "  " << q << "-qubit: " << sizes[q] << "\n";
        }
    }

    os << CYAN_FG << "=====================================\n" << RESET;
    return os;
}

std::ostream& CircuitGraph::displayFusionConfig(std::ostream& os) const {
    os << CYAN_FG << "======== Fusion Config: ========\n" << RESET;
    os << "max nqubits:          " << fusionConfig.maxNQubits << "\n";
    os << "max op count:         " << fusionConfig.maxOpCount;
    if (fusionConfig.maxOpCount < 0)
        os << " (infinite)";
    os << "\n";
    os << "zero skip thres:      " << std::scientific << fusionConfig.zeroSkippingThreshold << "\n";
    os << "allow multi traverse: " << ((fusionConfig.allowMultipleTraverse) ? "true" : "false") << "\n";
    os << "increment scheme:     " << ((fusionConfig.incrementScheme) ? "true" : "false") << "\n";
    os << CYAN_FG << "================================\n" << RESET;
    return os;
}

void CircuitGraph::dependencyAnalysis() {
    assert(false && "Not implemented yet!");
}

void CircuitGraph::greedyGateFusion() {
    if (tile.size() < 2)
        return;

    GateBlock* lhsBlock;
    GateBlock* rhsBlock;

    // print(std::cerr, 2) << "\n\n";
    int maxK = fusionConfig.maxNQubits;
    for (int currentK = (fusionConfig.incrementScheme) ? 2 : maxK;
                currentK <= maxK; currentK++) {
        fusionConfig.maxNQubits = currentK;
        bool hasChange = true;
        auto tileIt = tile.begin();
        unsigned q = 0;
        while (hasChange) {
            tileIt = tile.begin();
            hasChange = false;
            while (std::next(tileIt) != tile.end()) {
                // consecutive fuse
                q = 0;
                while (q < nqubits) {
                    if ((*tileIt)[q] == nullptr) {
                        q++;
                        continue;
                    }
                    if ((*std::next(tileIt))[q] == nullptr) {
                        auto downIt = repositionBlockDownward(tileIt, q);
                        q++;
                        continue;
                    }
                    auto* fusedBlock = tryFuseConnectedConsecutive(tileIt, q);
                    if (fusedBlock == nullptr)
                        q++;
                    else
                        hasChange = true;
                }
                // same row fuse
                q = 0;
                while (q < nqubits) {
                    auto* fusedBlock = tryFuseSameRow(tileIt, q);
                    if (fusedBlock == nullptr)
                        q++;
                }
                tileIt++;
            }
            eraseEmptyRows();
            updateTileUpward();
            if (!fusionConfig.allowMultipleTraverse)
                break;
        }
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
            if (it->lhsGate == nullptr) {
                std::cerr << RED_FG 
                          << "block " << id << " "
                          << "gate " << gate->id << " along qubit " << data.qubit << "\n" << RESET;
            }
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

void CircuitGraph::relabelBlocks() {
    currentBlockId = 0;
    auto allBlocks = getAllBlocks();
    for (auto* block : allBlocks)
        block->id = (currentBlockId++);
}
