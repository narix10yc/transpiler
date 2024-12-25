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

  std::cerr << "Trying to fuse "
            << "lhs " << lhs->id << " and rhs " << rhs->id << "\n";

  auto [benefit, cGate] = costModel->computeBenefit(
    *lhs->quantumGate, *rhs->quantumGate);

  if (benefit <= config.zeroTol)
    return nullptr;
  assert(cGate != nullptr);

  // fusion accepted
  auto* cBlock = graph.acquireGateBlock();
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

  cBlock->quantumGate = std::move(cGate);
  return cBlock;
}

using tile_iter_t = CircuitGraph::iter_t;

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
    for (const auto q : lBlock->quantumGate->qubits)
      (*tileIt)[q] = nullptr;
    for (const auto q : rBlock->quantumGate->qubits)
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

} // anonymous namespace

void saot::applyCPUGateFusion(
    const CPUFusionConfig& config, const CostModel* costModel,
    CircuitGraph& graph) {
  int nFused = 0;
  do {
    nFused = traverseAndFuse(config, costModel, graph);
    graph.squeeze();
  } while (config.multiTraverse && nFused > 0);
}
