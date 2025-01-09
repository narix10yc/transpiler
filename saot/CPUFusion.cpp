#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "utils/iocolor.h"

#include <list>

using namespace IOColor;
using namespace saot;

const CPUFusionConfig CPUFusionConfig::Disable {
  .zeroTol = 0.0,
  .multiTraverse = false,
  .incrementScheme = false,
  .benefitMargin = 0.0,
};

const CPUFusionConfig CPUFusionConfig::Minor {
  .zeroTol = 1e-8,
  .multiTraverse = false,
  .incrementScheme = true,
  .benefitMargin = 0.5,
};

const CPUFusionConfig CPUFusionConfig::Default {
  .zeroTol = 1e-8,
  .multiTraverse = true,
  .incrementScheme = true,
  .benefitMargin = 0.2,
};

const CPUFusionConfig CPUFusionConfig::Aggressive {
  .zeroTol = 1e-8,
  .multiTraverse = true,
  .incrementScheme = true,
  .benefitMargin = 0.0,
};

namespace {
GateBlock* computeCandidate(
    const CPUFusionConfig& config, const CostModel* costModel,
    CircuitGraph& graph,
    const GateBlock* lhs, const GateBlock* rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs == rhs)
    return nullptr;
  if (costModel == nullptr)
    return nullptr;

  assert(lhs->quantumGate != nullptr);
  assert(rhs->quantumGate != nullptr);

  // std::cerr << "Trying to fuse "
            // << "lhs " << lhs->id << " and rhs " << rhs->id << "\n";

  auto [benefit, cGate] = costModel->computeBenefit(
    *lhs->quantumGate, *rhs->quantumGate, graph.getContext());

  if (benefit <= config.zeroTol)
    return nullptr;
  assert(cGate != nullptr);

  // fusion accepted
  auto* cBlock = graph.acquireGateBlockForward();
  // set up connections
  for (const auto& lWire : lhs->wires) {
    auto qubit = lWire.qubit;
    GateNode* rhsEntry;
    if (auto *rhsWire = rhs->findWire(qubit); rhsWire == nullptr)
      rhsEntry = lWire.rhsEntry;
    else
      rhsEntry = rhsWire->rhsEntry;
    cBlock->wires.emplace_back(qubit, lWire.lhsEntry, rhsEntry);
  }

  for (const auto& rWire : rhs->wires) {
    auto qubit = rWire.qubit;
    if (lhs->findWire(qubit) == nullptr) {
      cBlock->wires.push_back(rWire);
    }
  }

  cBlock->quantumGate = cGate;
  return cBlock;
}

using tile_iter_t = CircuitGraph::tile_iter_t;

int getKAfterFusion(const GateBlock* blockA, const GateBlock* blockB) {
  int count = 0;
  auto itA = blockA->quantumGate->qubits.begin();
  auto itB = blockB->quantumGate->qubits.begin();
  const auto endA = blockA->quantumGate->qubits.end();
  const auto endB = blockB->quantumGate->qubits.end();
  while (itA != endA || itB != endB) {
    count++;
    if (itA == endA) {
      ++itB;
      continue;
    }
    if (itB == endB) {
      ++itA;
      continue;
    }
    if (*itA == *itB) {
      ++itA;
      ++itB;
      continue;
    }
    if (*itA > *itB) {
      ++itB;
      continue;
    }
    assert(*itA < *itB);
    ++itA;
    continue;
  }

  std::cerr << "getKAfterFusion " << blockA->id << " ";
  utils::printArray(std::cerr, blockA->quantumGate->qubits);
  std::cerr << " and " << blockB->id << " ";
  utils::printArray(std::cerr, blockB->quantumGate->qubits);
  std::cerr << " gives " << count << "\n";

  return count;
}

/// Fuse two blocks on the same row. \c graph will be updated.
/// Notice this function will NOT delete old blocks.
/// @return fused block
GateBlock* fuseAndInsertSameRow(
    CircuitGraph& graph, tile_iter_t iter,
    GateBlock* blockA, GateBlock* blockB) {
  auto* blockFused = graph.acquireGateBlock(blockA, blockB);
  for (const auto& wireA : blockA->wires)
    (*iter)[wireA.qubit] = blockFused;
  for (const auto& wireB : blockB->wires)
    (*iter)[wireB.qubit] = blockFused;

  graph.print(
    std::cerr << "SameRowFused " << blockA->id << " and "
              << blockB->id << " => " << blockFused->id << "\n", 2);
  return blockFused;
}

/// Fuse two blocks on different rows. \c graph will be updated.
/// Notice this function will NOT delete old blocks.
/// @return the tile iterator of the fused block
tile_iter_t fuseAndInsertDiffRow(
    CircuitGraph& graph, tile_iter_t iterL, int qubit) {
  assert(iterL != nullptr);
  auto* blockL = (*iterL)[qubit];
  assert(blockL != nullptr);
  auto iterR = iterL.next();
  assert(iterR != nullptr);
  auto* blockR = (*iterR)[qubit];
  assert(blockR);

  auto* blockFused = graph.acquireGateBlock(blockL, blockR);
  for (const auto& wireL : blockL->wires)
    (*iterL)[wireL.qubit] = nullptr;
  for (const auto& wireR : blockR->wires)
    (*iterR)[wireR.qubit] = nullptr;

  // prioritize iterR > iterL > between iterL and iterR
  if (graph.isRowVacant(*iterR, blockFused)) {
    for (const auto& wireF : blockFused->wires)
      (*iterR)[wireF.qubit] = blockFused;
    return iterR;
  }
  if (graph.isRowVacant(*iterL, blockFused)) {
    for (const auto& wireF : blockFused->wires)
      (*iterL)[wireF.qubit] = blockFused;
    return iterL;
  }
  auto iterInserted = graph.tile().emplace_insert(iterL);
  for (const auto& wireL : blockL->wires)
    (*iterInserted)[wireL.qubit] = blockFused;
  return iterInserted;
}

struct TentativeFusedItem {
  GateBlock* block;
  tile_iter_t iter;
};

/// @return Number of fused blocks
int startFusion(
    CircuitGraph& graph, const CostModel* costModel, const int maxK,
    tile_iter_t curIt, const int qubit) {
  auto* curBlock = (*curIt)[qubit];
  if (curBlock == nullptr)
    return 0;

  std::vector<TentativeFusedItem> fusedBlocks;
  fusedBlocks.reserve(8);
  fusedBlocks.emplace_back(curBlock, curIt);

  const auto checkFuseable = [&](GateBlock* candidateBlock) {
    if (candidateBlock == nullptr)
      return false;
    if (std::ranges::find_if(fusedBlocks,
        [b=candidateBlock](const auto& item) {
          return item.block == b;
        }) != fusedBlocks.end()) {
      return false;
    }
    return getKAfterFusion(curBlock, candidateBlock) <= maxK;
  };

  // Start with same-row blocks
  for (int q = qubit+1; q < graph.nqubits; ++q) {
    auto* candidateBlock = (*curIt)[q];
    if (checkFuseable(candidateBlock) == false)
      continue;

    // candidateBlock is accepted
    fusedBlocks.emplace_back(candidateBlock, curIt);
    curBlock = fuseAndInsertSameRow(graph, curIt, curBlock, candidateBlock);
  }

  std::cerr << "Same row done\n";
  bool progress;
  do {
    ++curIt;
    if (curIt == graph.tile_end())
      break;
    progress = false;
    std::cerr << "Try fusing with row @ " << &(*curIt) << "\n";
    for (const auto& q : curBlock->quantumGate->qubits) {
      std::cerr << "qubit " << q << "\n";
      auto* candidateBlock = (*curIt)[q];
      if (checkFuseable(candidateBlock) == false)
        continue;
      // candidateBlock is accepted
      fusedBlocks.emplace_back(candidateBlock, curIt);
      curIt = fuseAndInsertDiffRow(graph, curIt.prev(), q);
      curBlock = (*curIt)[candidateBlock->quantumGate->qubits[0]];
      std::cerr << "curIt is now " << &(*curIt) << "\n"
                << "curBlock is now " << curBlock->id << "\n";
      progress = true;
      graph.print(std::cerr, 2) << "\n";
      break;
    }
  } while (progress == true);

  // Check benefit
  std::cerr << "[\n";
  for (const auto& tentative : fusedBlocks)
    std::cerr << tentative.block->id << " @ row " << &(*tentative.iter) << "\n";
  std::cerr << "]\n";

  return fusedBlocks.size();
}

// GateBlock* trySameWireFuse(const CPUFusionConfig& config, CircuitGraph& graph,
//                            const tile_iter_t& itLHS, int q_) {
//   assert(itLHS != graph.tile().end());
//   const auto itRHS = std::next(itLHS);
//   if (itRHS == graph.tile().end())
//     return nullptr;
//
//   GateBlock* lhs = (*itLHS)[q_];
//   GateBlock* rhs = (*itRHS)[q_];
//
//   if (!lhs || !rhs)
//     return nullptr;
//
//   GateBlock* block = computeCandidate(lhs, rhs, config);
//   if (block == nullptr)
//     return nullptr;
//
//   // std::cerr << BLUE_FG;
//   // lhs->displayInfo(std::cerr);
//   // rhs->displayInfo(std::cerr);
//   // block->displayInfo(std::cerr) << RESET;
//
//   for (const auto& q : lhs->getQubits())
//     (*itLHS)[q] = nullptr;
//   for (const auto& q : rhs->getQubits())
//     (*itRHS)[q] = nullptr;
//
//   delete (lhs);
//   delete (rhs);
//
//   // insert block to tile
//   graph.insertBlock(itLHS, block);
//   return block;
// }

/// Try to fuse block
/// \code (*tileIt)[q]        \endcode and
/// \code (*tileIt.next())[q] \endcode.
/// It will check fusion eligibility using \c config.
/// If fusion is accepted, delete old blocks and append fused block into the
/// graph, and return the fused block.
int tryCrossWireFuse(
    const CPUFusionConfig& config, const CostModel* costModel,
    CircuitGraph& graph, tile_iter_t tileIt) {
  int nFused = 0;
  auto tileNext = tileIt.next();
  if (tileNext == nullptr)
    return 0;

  for (unsigned q = 0; q < graph.nqubits; ++q) {
    auto* lBlock = (*tileIt)[q];
    auto* rBlock = (*tileNext)[q];
    auto* cBlock = computeCandidate(config, costModel, graph, lBlock, rBlock);
    if (cBlock == nullptr)
      continue;

    // fusion accepted
    for (const auto& q : lBlock->quantumGate->qubits)
      (*tileIt)[q] = nullptr;
    for (const auto& q : rBlock->quantumGate->qubits)
      (*tileNext)[q] = nullptr;

    auto insertedIt = graph.insertBlock(tileIt, cBlock);
    graph.releaseGateBlock(lBlock);
    graph.releaseGateBlock(rBlock);
    ++nFused;
    // terminate if a new row is inserted (such cases should be handled in the
    // next traversal
    if (insertedIt != tileIt && insertedIt != tileNext)
      break;
  }
  return nFused;
}

/// @return Number of fused blocks in this traversal
int traverseAndFuse(
    const CPUFusionConfig& config, const CostModel* costModel,
    CircuitGraph& graph) {
  int nFused = 0;
  auto it = graph.tile().begin();
  while (it != nullptr) {
    nFused += tryCrossWireFuse(config, costModel, graph, it);
    ++it;
  }
  return nFused;
}

GateBlock* computeTwoQubitCandidate(
    CircuitGraph& graph, GateBlock* lBlock, GateBlock* rBlock) {
  if (lBlock == nullptr || rBlock == nullptr)
    return nullptr;

  const auto lNQubits = lBlock->nqubits();
  const auto rNQubits = rBlock->nqubits();
  assert(lNQubits == 1 || lNQubits == 2);
  assert(rNQubits == 1 || rNQubits == 2);

  const auto& lWires = lBlock->wires;
  const auto& rWires = rBlock->wires;

  const auto l0 = lWires[0].qubit;
  const auto r0 = rWires[0].qubit;
  const auto l1 = (lNQubits == 2) ? lWires[1].qubit : -1;
  const auto r1 = (rNQubits == 2) ? rWires[1].qubit : -1;

  if (lNQubits == 2 && rNQubits == 2 && (l0 != r0 || l1 != r1))
    return nullptr;

  auto* quantumGate = graph.acquireQuantumGateForward(
    rBlock->quantumGate->lmatmul(*lBlock->quantumGate));
  auto* cBlock = graph.acquireGateBlockForward();
  cBlock->quantumGate = quantumGate;

  if (lNQubits == 1 && rNQubits == 1) {
    assert(l0 == r0);
    cBlock->wires.emplace_back(l0, lWires[0].lhsEntry, rWires[0].rhsEntry);
    return cBlock;
  }

  if (lNQubits == 2 && rNQubits == 2) {
    if (l0 != r0 || l1 != r1) {
      graph.releaseGateBlock(cBlock);
      return nullptr;
    }
    cBlock->wires.emplace_back(l0, lWires[0].lhsEntry, rWires[0].rhsEntry);
    cBlock->wires.emplace_back(l1, lWires[1].lhsEntry, rWires[1].rhsEntry);
    return cBlock;
  }

  if (lNQubits == 1 && rNQubits == 2) {
    if (l0 == r0) {
      cBlock->wires.emplace_back(l0, lWires[0].lhsEntry, rWires[0].rhsEntry);
      cBlock->wires.emplace_back(r1, rWires[1].lhsEntry, rWires[1].rhsEntry);
    } else {
      assert(l0 == r1);
      cBlock->wires.emplace_back(r0, rWires[0].lhsEntry, rWires[0].rhsEntry);
      cBlock->wires.emplace_back(r1, lWires[0].lhsEntry, rWires[1].rhsEntry);
    }
    return cBlock;
  }

  assert(lNQubits == 2 && rNQubits == 1);
  if (l0 == r0) {
    cBlock->wires.emplace_back(l0, lWires[0].lhsEntry, rWires[0].rhsEntry);
    cBlock->wires.emplace_back(r1, lWires[1].lhsEntry, lWires[1].rhsEntry);
  } else {
    assert(l1 == r0);
    cBlock->wires.emplace_back(r0, lWires[0].lhsEntry, lWires[0].rhsEntry);
    cBlock->wires.emplace_back(r1, lWires[1].lhsEntry, rWires[0].rhsEntry);
  }
  return cBlock;
}

int applyTwoQubitFusion(CircuitGraph& graph) {
  int nFused = 0;
  while (true) {
    int nFusedThisTraversal = 0;
    auto tileIt = graph.tile().begin();
    auto tileNext = tileIt.next();
    auto tileEnd = graph.tile().end();
    while (tileNext != tileEnd) {
      for (unsigned q = 0; q < graph.nqubits; ++q) {
        auto* lBlock = (*tileIt)[q];
        auto* rBlock = (*tileNext)[q];
        if (auto* cBlock = computeTwoQubitCandidate(graph, lBlock, rBlock)) {
          for (const auto& q : lBlock->quantumGate->qubits)
            (*tileIt)[q] = nullptr;
          for (const auto& q : rBlock->quantumGate->qubits)
            (*tileNext)[q] = nullptr;
          graph.insertBlock(tileIt, cBlock);
          graph.releaseGateBlock(lBlock);
          graph.releaseGateBlock(rBlock);
          ++nFusedThisTraversal;
        }
      }
      ++tileIt;
      ++tileNext;
    }

    if (nFusedThisTraversal > 0) {
      graph.squeeze();
      nFused += nFusedThisTraversal;
      continue;
    }
    break;
  }
  return nFused;
}

} // anonymous namespace

void saot::applyCPUGateFusion(
    const CPUFusionConfig& config, const CostModel* costModel,
    CircuitGraph& graph) {
  startFusion(graph, costModel, 3, graph.tile_begin(), 0);
  // applyTwoQubitFusion(graph);
  // int nFused = 0;
  // do {
    // nFused = traverseAndFuse(config, costModel, graph);
    // graph.squeeze();
  // } while (config.multiTraverse && nFused > 0);
}
