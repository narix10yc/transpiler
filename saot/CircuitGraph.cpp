#include "saot/CircuitGraph.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <chrono>
#include <iomanip>
#include <map>
#include <numeric>
#include <thread>

using namespace IOColor;
using namespace saot;

// static member
int GateNode::idCount = 0;
int GateBlock::idCount = 0;

GateNode::GateNode(
    std::unique_ptr<QuantumGate> _gate, const CircuitGraph &graph)
    : id(idCount++), quantumGate(std::move(_gate)) {
  connections.reserve(quantumGate->nqubits());
  for (const auto q : quantumGate->qubits) {
    connections.emplace_back(q, nullptr, nullptr);
    auto it = graph.tile().tail_iter();
    while (it != nullptr && (*it)[q] == nullptr)
      --it;
    if (it != nullptr) {
      auto* lhsBlockWire = (*it)[q]->findWire(q);
      assert(lhsBlockWire != nullptr);
      lhsBlockWire->rhsEntry->connect(this, q);
    }
  }
}

void GateNode::connect(GateNode* rhsGate, int q) {
  assert(rhsGate);
  auto* myIt = findConnection(q);
  assert(myIt);
  auto* rhsIt = rhsGate->findConnection(q);
  assert(rhsIt);

  myIt->rhsGate = rhsGate;
  rhsIt->lhsGate = this;
}

//
// CircuitGraph CircuitGraph::QFTCircuit(int nqubits) {
//   CircuitGraph graph;
//   for (int q = 0; q < nqubits; ++q) {
//     graph.addGate(std::make_unique<QuantumGate>(GateMatrix::MatrixH_c, q));
//     for (int l = q + 1; l < nqubits; ++l) {
//       double angle = M_PI_2 * std::pow(2.0, q - l);
//       graph.addGate(std::make_unique<QuantumGate>(
//         GateMatrix::FromName("cp", {angle}), std::initializer_list<int>{q, l}));
//     }
//   }
//   return graph;
// }
//
// CircuitGraph CircuitGraph::ALACircuit(int nqubits, int nrounds) {
//   assert(0 && "Not Implemented");
//   CircuitGraph graph;
//   return graph;
// }

// CircuitGraph CircuitGraph::GetTestCircuit(
//     const GateMatrix& gateMatrix, int nqubits, int nrounds) {
//   CircuitGraph graph;
//   auto nqubitsGate = gateMatrix.nqubits();
//
//   for (int r = 0; r < nrounds; r++) {
//     for (int q = 0; q < nqubits; q++) {
//       graph.addGate(std::make_unique<QuantumGate>(
//         gateMatrix,
//         std::initializer_list<int>{q, (q + 1) % nqubits, (q + 2) % nqubits});
//     }
//   }
//   return graph;
// }

CircuitGraph::iter_t CircuitGraph::insertBlock(iter_t it, GateBlock* block) {
  assert(it != nullptr);
  assert(block != nullptr);

  const auto& qubits = block->quantumGate->qubits;
  assert(!qubits.empty());

  // try insert to current row
  if (isRowVacant(*it, block)) {
    for (const auto& q : qubits)
      (*it)[q] = block;
    return it;
  }

  // try insert to next row
  it = it.next();
  if (it != nullptr && isRowVacant(*it, block)) {
    for (const auto& q : qubits)
      (*it)[q] = block;
    return it;
  }

  // insert between current and next row
  it = _tile.emplace_insert(it);
  for (const auto& q : qubits)
    (*it)[q] = block;
  return it;
}
//
// void CircuitGraph::updateBlockConnections(iter_t it, int q) {
//   auto* block = (*it)[q];
//   assert(block != nullptr && "Trying to update connection of a NULL block?");
//   auto prevIt = it.prev();
//   while (prevIt != nullptr && (*prevIt)[q] == nullptr)
//     --prevIt;
//   auto nextIt = it.next();
//   while (nextIt != nullptr && (*nextIt)[q] == nullptr)
//     ++nextIt;
//
//   for (auto& wire : block->wires) {
//     auto qubit = wire.qubit;
//     // left connection
//     if (prevIt) {
//       wire.lhsEntry = (*prevIt)[q]->findWire(q)->rhsEntry;
//     } else
//       wire.lhsEntry = nullptr;
//   }
//
//
// }


void CircuitGraph::appendGate(std::unique_ptr<QuantumGate> quantumGate) {
  assert(quantumGate != nullptr);
  // update nqubits
  for (const auto& q : quantumGate->qubits) {
    if (q >= nqubits)
      nqubits = q + 1;
  }

  // create gate and setup connections
  auto* gate = gateNodePool.acquire(std::move(quantumGate), *this);

  // create block and insert to the tile
  // TODO: this is slightly inefficient as the block may be assigned twice
  auto* block = gateBlockPool.acquire(gate);
  auto it = insertBlock(iter_t(_tile.tail()), block);
  repositionBlockUpward(it, block->quantumGate->qubits[0]);
}

// std::vector<GateBlock*> CircuitGraph::getAllBlocks() const {
//   std::vector<GateBlock*> allBlocks;
//   std::vector<GateBlock*> rowBlocks;
//   for (const auto& row : _tile) {
//     rowBlocks.clear();
//     for (const auto& block : row) {
//       if (block == nullptr)
//         continue;
//       if (std::find(rowBlocks.begin(), rowBlocks.end(), block) ==
//           rowBlocks.end())
//         rowBlocks.push_back(block);
//     }
//     for (const auto& block : rowBlocks)
//       allBlocks.push_back(block);
//   }
//   return allBlocks;
// }
//
CircuitGraph::list_node_t*
CircuitGraph::repositionBlockUpward(list_node_t* ln, int q) {
  assert(ln != nullptr);
  auto* block = ln->data[q];
  assert(block && "Cannot reposition a null block");
  // find which row fits the block
  auto* newln = ln;
  const auto* const head = _tile.head();
  if (newln == head)
    return newln;

  do {
    if (isRowVacant(newln->prev->data, block))
      newln = newln->prev;
    else
      break;
  } while (newln != head);

  // put block into the new position
  if (newln == ln)
    return ln;
  for (const auto& data : block->wires) {
    const auto& i = data.qubit;
    ln->data[i] = nullptr;
    newln->data[i] = block;
  }

  return newln;
}

CircuitGraph::list_node_t*
CircuitGraph::repositionBlockDownward(list_node_t* ln, int q) {
  assert(ln != nullptr);
  auto* block = ln->data[q];
  assert(block && "Cannot reposition a null block");
  // find which row fits the block
  auto* newln = ln;
  const auto* const tail = _tile.tail();
  if (newln == tail)
    return newln;

  do {
    if (isRowVacant(newln->next->data, block))
      newln = newln->next;
    else
      break;
  } while (newln != tail);

  // put block into the new position
  if (newln == ln)
    return ln;
  for (const auto& data : block->wires) {
    const auto& i = data.qubit;
    ln->data[i] = nullptr;
    newln->data[i] = block;
  }

  return newln;
}

void CircuitGraph::eraseEmptyRows() {
  auto it = _tile.cbegin();
  while (it != nullptr) {
    bool empty = true;
    for (unsigned q = 0; q < nqubits; q++) {
      if ((*it)[q] != nullptr) {
        empty = false;
        break;
      }
    }
    if (empty)
      it = _tile.erase(it);
    else
      ++it;
  }
}

void CircuitGraph::squeeze() {
  eraseEmptyRows();
  auto it = _tile.begin();
  while (it != nullptr) {
    for (unsigned q = 0; q < nqubits; q++) {
      if ((*it)[q])
        repositionBlockUpward(it, q);
    }
    ++it;
  }
  eraseEmptyRows();
}

std::ostream& CircuitGraph::print(std::ostream& os, int verbose) const {
  if (_tile.empty())
    return os << "<empty tile>\n";
  int width = static_cast<int>(std::log10(GateBlock::getIdCount()) + 1);
  if ((width & 1) == 0)
    width++;

  const std::string vbar =
      std::string(width / 2, ' ') + "|" + std::string(width / 2 + 1, ' ');

  for (const auto& row : _tile) {
    if (verbose > 1)
      os << &row << ": ";
    for (unsigned q = 0; q < nqubits; q++) {
      if (const auto* block = row[q]; block != nullptr)
        os << std::setw(width) << std::setfill('0') << block->id << " ";
      else
        os << vbar;
    }
    os << "\n";
  }
  return os;
}
//
// std::ostream& GateBlock::displayInfo(std::ostream& os) const {
//   os << "Block " << id << ": [";
//   for (const auto& data : wires) {
//     os << "(" << data.qubit << ":";
//     GateNode* gate = data.lhsEntry;
//     assert(gate);
//     os << gate->id << ",";
//     while (gate != data.rhsEntry) {
//       gate = gate->findRHS(data.qubit);
//       assert(gate);
//       os << gate->id << ",";
//     }
//     os << "),";
//   }
//   return os << "]\n";
// }
//
// std::vector<int> CircuitGraph::getBlockSizes() const {
//   std::vector<int> sizes(nqubits + 1, 0);
//   const auto allBlocks = getAllBlocks();
//   int largestSize = 0;
//   for (const auto* b : allBlocks) {
//     auto blockNQubits = b->nqubits();
//     sizes[blockNQubits]++;
//     if (blockNQubits > largestSize)
//       largestSize = blockNQubits;
//   }
//   sizes.resize(largestSize + 1);
//   return sizes;
// }
//
// std::vector<std::vector<int>> CircuitGraph::getBlockOpCountHistogram() const {
//   const auto allBlocks = getAllBlocks();
//   int largestSize = 0;
//   for (const auto* b : allBlocks) {
//     auto blockNQubits = b->nqubits();
//     if (blockNQubits > largestSize)
//       largestSize = blockNQubits;
//   }
//   std::vector<std::vector<int>> hist(largestSize + 1);
//   for (unsigned q = 1; q < largestSize + 1; q++)
//     hist[q].resize(q, 0);
//
//   for (const auto* b : allBlocks) {
//     const int q = b->nqubits();
//     int catagory = 0;
//     int opCount = b->quantumGate->opCount();
//     while ((1 << (2 * catagory + 3)) < opCount)
//       catagory++;
//
//     hist[q][catagory]++;
//   }
//   return hist;
// }
//
// std::ostream& CircuitGraph::displayInfo(std::ostream& os, int verbose) const {
//   os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n"
//      << RESET;
//
//   os << "- Number of Gates:  " << countGates() << "\n";
//   const auto allBlocks = getAllBlocks();
//   auto nBlocks = allBlocks.size();
//   os << "- Number of Blocks: " << nBlocks << "\n";
//   auto totalOp = countTotalOps();
//   os << "- Total Op Count:   " << totalOp << "\n";
//   os << "- Avergae Op Count: " << std::fixed << std::setprecision(1)
//      << static_cast<double>(totalOp) / nBlocks << "\n";
//   os << "- Circuit Depth:    " << _tile.size() << "\n";
//
//   if (verbose > 3) {
//     os << "- Block Sizes Count:\n";
//     std::vector<std::vector<int>> vec(nqubits + 1);
//     const auto allBlocks = getAllBlocks();
//     for (const auto* block : allBlocks)
//       vec[block->nqubits()].push_back(block->id);
//
//     for (unsigned q = 1; q < vec.size(); q++) {
//       if (vec[q].empty())
//         continue;
//       os << "  " << q << "-qubit: count " << vec[q].size() << " ";
//       utils::printVector(vec[q], os) << "\n";
//     }
//   } else if (verbose > 2) {
//     os << "- Block Statistics:\n";
//     const auto hist = getBlockOpCountHistogram();
//     for (unsigned q = 1; q < hist.size(); q++) {
//       auto count = std::reduce(hist[q].begin(), hist[q].end());
//       if (count == 0)
//         continue;
//       os << "  " << q << "-qubit count = " << count << "; hist: ";
//       utils::printVector(hist[q], os) << "\n";
//     }
//   } else if (verbose > 1) {
//     os << "- Block Sizes Count:\n";
//     const auto sizes = getBlockSizes();
//     for (unsigned q = 1; q < sizes.size(); q++) {
//       if (sizes[q] <= 0)
//         continue;
//       os << "  " << q << "-qubit: " << sizes[q] << "\n";
//     }
//   }
//
//   os << CYAN_FG << "=====================================\n" << RESET;
//   return os;
// }
//
// std::vector<GateNode*> GateBlock::getOrderedGates() const {
//   std::deque<GateNode*> queue;
//   // vector should be more efficient as we expect small sizes here
//   std::vector<GateNode*> gates;
//   for (const auto& data : wires) {
//     if (std::ranges::find(queue, data.rhsEntry) == queue.end())
//       queue.push_back(data.rhsEntry);
//   }
//
//   while (!queue.empty()) {
//     const auto& gate = queue.back();
//     if (std::ranges::find(gates, gate) != gates.end()) {
//       queue.pop_back();
//       continue;
//     }
//     std::vector<GateNode*> higherPriorityGates;
//     for (const auto& data : this->wires) {
//       if (gate == data.lhsEntry)
//         continue;
//       auto it = gate->findConnection(data.qubit);
//       if (it == nullptr)
//         continue;
//       if (it->lhsGate == nullptr) {
//         std::cerr << RED_FG << "block " << id << " "
//                   << "gate " << gate->id << " along qubit " << data.qubit
//                   << "\n"
//                   << RESET;
//       }
//       assert(it->lhsGate);
//       if (std::find(gates.begin(), gates.end(), it->lhsGate) == gates.end())
//         higherPriorityGates.push_back(it->lhsGate);
//     }
//
//     if (higherPriorityGates.empty()) {
//       queue.pop_back();
//       gates.push_back(gate);
//     } else {
//       for (const auto& g : higherPriorityGates)
//         queue.push_back(g);
//     }
//   }
//   return gates;
// }
//
// void CircuitGraph::relabelBlocks() const {
//   int count = 0;
//   auto allBlocks = getAllBlocks();
//   for (auto* block : allBlocks)
//     block->id = (count++);
// }
