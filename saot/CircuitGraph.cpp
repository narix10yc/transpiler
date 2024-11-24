#include "saot/CircuitGraph.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <chrono>
#include <deque>
#include <iomanip>
#include <map>
#include <numeric>
#include <thread>

using namespace IOColor;
using namespace saot;

// static member
int GateNode::idCount = 0;
int GateBlock::idCount = 0;

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
  for (auto &data : dataVector) {
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
  for (auto &data : dataVector) {
    auto rhsIt = rhsBlock->findQubit(data.qubit);
    if (rhsIt == rhsBlock->dataVector.end())
      continue;
    data.rhsEntry->connect(rhsIt->lhsEntry, data.qubit);
    count++;
  }
  return count;
}

CircuitGraph CircuitGraph::QFTCircuit(int nqubits) {
  CircuitGraph graph;
  for (int q = 0; q < nqubits; ++q) {
    graph.addGate(GateMatrix(GateMatrix::MatrixH_c), {q});
    for (int l = q + 1; l < nqubits; ++l) {
      double angle = M_PI_2 * std::pow(2.0, q - l);
      graph.addGate(GateMatrix::FromName("cp", {angle}), {q, l});
    }
  }
  return graph;
}

CircuitGraph CircuitGraph::ALACircuit(int nqubits, int nrounds) {
  assert(0 && "Not Implemented");
  CircuitGraph graph;
  return graph;
}

CircuitGraph CircuitGraph::GetTestCircuit(const GateMatrix &gateMatrix,
                                          int nqubits, int nrounds) {
  CircuitGraph graph;
  auto nqubitsGate = gateMatrix.nqubits();

  for (int r = 0; r < nrounds; r++) {
    for (int q = 0; q < nqubits; q++) {
      graph.addGate(
          QuantumGate(gateMatrix, {q, (q + 1) % nqubits, (q + 2) % nqubits}));
    }
  }
  return graph;
}

CircuitGraph::tile_iter_t CircuitGraph::insertBlock(tile_iter_t it,
                                                    GateBlock* block) {
  assert(block);

  const auto qubits = block->getQubits();
  assert(!qubits.empty());

  // try insert to current row
  if (isRowVacant(it, block)) {
    // insert at it
    for (const auto &q : qubits) {
      assert((*it)[q] == nullptr);
      (*it)[q] = block;
    }
    return it;
  }

  // try insert to next row
  it++;
  if (it == _tile.end()) {
    row_t row{nullptr};
    for (const auto &q : qubits)
      row[q] = block;
    return _tile.insert(it, row);
  }
  if (isRowVacant(it, block)) {
    // insert at it
    for (const auto &q : qubits) {
      assert((*it)[q] == nullptr);
      (*it)[q] = block;
    }
    return it;
  }

  // insert to between current and next row
  row_t row{nullptr};
  for (const auto &q : qubits)
    row[q] = block;
  return _tile.insert(it, row);
}

void CircuitGraph::addGate(const GateMatrix &matrix,
                           const std::vector<int> &qubits) {
  assert(matrix.nqubits() == qubits.size());

  // update nqubits
  for (const auto &q : qubits) {
    if (q >= nqubits)
      nqubits = q + 1;
  }

  // create gate and block
  auto gate = new GateNode(matrix, qubits);
  auto block = new GateBlock(gate);

  // insert block to tile
  auto rit = _tile.rbegin();
  while (rit != _tile.rend()) {
    bool vacant = true;
    for (const auto &q : qubits) {
      if ((*rit)[q] != nullptr) {
        vacant = false;
        break;
      }
    }
    if (!vacant)
      break;
    rit++;
  }
  if (rit == _tile.rbegin())
    _tile.push_back({nullptr});
  else
    rit--;
  for (const auto &q : qubits)
    (*rit)[q] = block;

  // set up gate and block connections
  auto itRHS = --rit.base();
  if (itRHS == _tile.begin())
    return;

  tile_iter_t itLHS;
  GateBlock* lhsBlock;
  for (const auto &q : qubits) {
    itLHS = itRHS;
    while (--itLHS != _tile.begin()) {
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
  for (const auto &row : _tile) {
    rowBlocks.clear();
    for (const auto &block : row) {
      if (block == nullptr)
        continue;
      if (std::find(rowBlocks.begin(), rowBlocks.end(), block) ==
          rowBlocks.end())
        rowBlocks.push_back(block);
    }
    for (const auto &block : rowBlocks)
      allBlocks.push_back(block);
  }
  return allBlocks;
}

CircuitGraph::tile_iter_t CircuitGraph::repositionBlockUpward(tile_iter_t it,
                                                              size_t q_) {
  auto* block = (*it)[q_];
  assert(block);

  if (it == _tile.begin())
    return it;

  bool vacant = true;
  auto newIt = it;
  while (true) {
    newIt--;
    vacant = true;
    for (const auto &data : block->dataVector) {
      if ((*newIt)[data.qubit] != nullptr) {
        vacant = false;
        break;
      }
    }
    if (!vacant) {
      newIt++;
      break;
    }
    if (newIt == _tile.begin())
      break;
  }

  if (newIt == it)
    return newIt;
  for (const auto &data : block->dataVector) {
    const auto &q = data.qubit;
    (*it)[q] = nullptr;
    (*newIt)[q] = block;
  }
  return newIt;
}

CircuitGraph::tile_riter_t
CircuitGraph::repositionBlockDownward(tile_riter_t it, size_t q_) {
  auto* block = (*it)[q_];
  assert(block);

  if (it == _tile.rbegin())
    return it;

  bool vacant = true;
  auto newIt = it;
  while (true) {
    newIt--;
    vacant = true;
    for (const auto &data : block->dataVector) {
      if ((*newIt)[data.qubit] != nullptr) {
        vacant = false;
        break;
      }
    }
    if (!vacant) {
      newIt++;
      break;
    }
    if (newIt == _tile.rbegin())
      break;
  }

  if (newIt == it)
    return newIt;
  for (const auto &data : block->dataVector) {
    const auto &q = data.qubit;
    (*it)[q] = nullptr;
    (*newIt)[q] = block;
  }
  return newIt;
}

void CircuitGraph::eraseEmptyRows() {
  auto it = _tile.cbegin();
  bool empty;
  while (it != _tile.cend()) {
    empty = true;
    for (unsigned q = 0; q < nqubits; q++) {
      if ((*it)[q]) {
        empty = false;
        break;
      }
    }
    if (empty)
      it = _tile.erase(it);
    else
      it++;
  }
}

void CircuitGraph::updateTileUpward() {
  eraseEmptyRows();
  auto it = _tile.begin();
  while (it != _tile.end()) {
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

  auto it = _tile.rbegin();
  while (it != _tile.rend()) {
    for (unsigned q = 0; q < nqubits; q++) {
      if ((*it)[q])
        repositionBlockDownward(it, q);
    }
    it++;
  }
  eraseEmptyRows();
}

std::ostream &CircuitGraph::print(std::ostream &os, int verbose) const {
  auto nBlocks = countBlocks();
  if (nBlocks == 0)
    return os << "<empty tile>\n";
  int width = static_cast<int>(std::log10(nBlocks) + 1);
  if ((width & 1) == 0)
    width++;

  std::string vbar =
      std::string(width / 2, ' ') + "|" + std::string(width / 2 + 1, ' ');

  auto it = _tile.cbegin();
  while (it != _tile.cend()) {
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

std::ostream &GateBlock::displayInfo(std::ostream &os) const {
  os << "Block " << id << ": [";
  for (const auto &data : dataVector) {
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
    auto blockNQubits = b->nqubits();
    sizes[blockNQubits]++;
    if (blockNQubits > largestSize)
      largestSize = blockNQubits;
  }
  sizes.resize(largestSize + 1);
  return sizes;
}

std::vector<std::vector<int>> CircuitGraph::getBlockOpCountHistogram() const {
  const auto allBlocks = getAllBlocks();
  int largestSize = 0;
  for (const auto* b : allBlocks) {
    auto blockNQubits = b->nqubits();
    if (blockNQubits > largestSize)
      largestSize = blockNQubits;
  }
  std::vector<std::vector<int>> hist(largestSize + 1);
  for (unsigned q = 1; q < largestSize + 1; q++)
    hist[q].resize(q, 0);

  for (const auto* b : allBlocks) {
    const int q = b->nqubits();
    int catagory = 0;
    int opCount = b->quantumGate->opCount();
    while ((1 << (2 * catagory + 3)) < opCount)
      catagory++;

    hist[q][catagory]++;
  }
  return hist;
}

std::ostream &CircuitGraph::displayInfo(std::ostream &os, int verbose) const {
  os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n"
     << RESET;

  os << "- Number of Gates:  " << countGates() << "\n";
  const auto allBlocks = getAllBlocks();
  auto nBlocks = allBlocks.size();
  os << "- Number of Blocks: " << nBlocks << "\n";
  auto totalOp = countTotalOps();
  os << "- Total Op Count:   " << totalOp << "\n";
  os << "- Avergae Op Count: " << std::fixed << std::setprecision(1)
     << static_cast<double>(totalOp) / nBlocks << "\n";
  os << "- Circuit Depth:    " << _tile.size() << "\n";

  if (verbose > 3) {
    os << "- Block Sizes Count:\n";
    std::vector<std::vector<int>> vec(nqubits + 1);
    const auto allBlocks = getAllBlocks();
    for (const auto* block : allBlocks)
      vec[block->nqubits()].push_back(block->id);

    for (unsigned q = 1; q < vec.size(); q++) {
      if (vec[q].empty())
        continue;
      os << "  " << q << "-qubit: count " << vec[q].size() << " ";
      utils::printVector(vec[q], os) << "\n";
    }
  } else if (verbose > 2) {
    os << "- Block Statistics:\n";
    const auto hist = getBlockOpCountHistogram();
    for (unsigned q = 1; q < hist.size(); q++) {
      auto count = std::reduce(hist[q].begin(), hist[q].end());
      if (count == 0)
        continue;
      os << "  " << q << "-qubit count = " << count << "; hist: ";
      utils::printVector(hist[q], os) << "\n";
    }
  } else if (verbose > 1) {
    os << "- Block Sizes Count:\n";
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

std::vector<GateNode*> GateBlock::getOrderedGates() const {
  std::deque<GateNode*> queue;
  // vector should be more efficient as we expect small size here
  std::vector<GateNode*> gates;
  for (const auto &data : dataVector) {
    if (std::find(queue.begin(), queue.end(), data.rhsEntry) == queue.end())
      queue.push_back(data.rhsEntry);
  }

  while (!queue.empty()) {
    const auto &gate = queue.back();
    if (std::find(gates.begin(), gates.end(), gate) != gates.end()) {
      queue.pop_back();
      continue;
    }
    std::vector<GateNode*> higherPriorityGates;
    for (const auto &data : this->dataVector) {
      if (gate == data.lhsEntry)
        continue;
      auto it = gate->findQubit(data.qubit);
      if (it == gate->dataVector.end())
        continue;
      if (it->lhsGate == nullptr) {
        std::cerr << RED_FG << "block " << id << " "
                  << "gate " << gate->id << " along qubit " << data.qubit
                  << "\n"
                  << RESET;
      }
      assert(it->lhsGate);
      if (std::find(gates.begin(), gates.end(), it->lhsGate) == gates.end())
        higherPriorityGates.push_back(it->lhsGate);
    }

    if (higherPriorityGates.empty()) {
      queue.pop_back();
      gates.push_back(gate);
    } else {
      for (const auto &g : higherPriorityGates)
        queue.push_back(g);
    }
  }
  return gates;
}

void CircuitGraph::relabelBlocks() {
  int count = 0;
  auto allBlocks = getAllBlocks();
  for (auto* block : allBlocks)
    block->id = (count++);
}
