#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "utils/iocolor.h"

#include <list>

using namespace IOColor;
using namespace saot;

CPUFusionConfig CPUFusionConfig::Disable =
    CPUFusionConfig{.maxNQubits = 0,
                    .maxOpCount = 0,
                    .zeroSkippingThreshold = 0.0,
                    .allowMultipleTraverse = true,
                    .incrementScheme = true};

CPUFusionConfig CPUFusionConfig::TwoQubitOnly =
    CPUFusionConfig{.maxNQubits = 2,
                    .maxOpCount = 64, // 2-qubit dense
                    .zeroSkippingThreshold = 1e-8,
                    .allowMultipleTraverse = true,
                    .incrementScheme = true};

CPUFusionConfig CPUFusionConfig::Default =
    CPUFusionConfig{.maxNQubits = 5,
                    .maxOpCount = 256, // 3-qubit dense
                    .zeroSkippingThreshold = 1e-8,
                    .allowMultipleTraverse = true,
                    .incrementScheme = true};

CPUFusionConfig CPUFusionConfig::Aggressive =
    CPUFusionConfig{.maxNQubits = 7,
                    .maxOpCount = 4096, // 5.5-qubit dense
                    .zeroSkippingThreshold = 1e-8,
                    .allowMultipleTraverse = true,
                    .incrementScheme = true};

std::ostream& CPUFusionConfig::display(std::ostream& os) const {
  os << CYAN("======== Fusion Config: ========\n");
  os << "max nqubits:          " << maxNQubits << "\n";
  os << "max op count:         " << maxOpCount;
  if (maxOpCount < 0)
    os << " (infinite)";
  os << "\n";
  os << "zero skip thres:      " << std::scientific << zeroSkippingThreshold
     << "\n";
  os << "allow multi traverse: " << ((allowMultipleTraverse) ? "true" : "false")
     << "\n";
  os << "increment scheme:     " << ((incrementScheme) ? "true" : "false")
     << "\n";
  os << CYAN("================================\n");
  return os;
}

namespace {
GateBlock* computeCandidate(
    const CPUFusionConfig& config, CircuitGraph& graph,
    const GateBlock* lhs, const GateBlock* rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs == rhs)
    return nullptr;

  assert(lhs->quantumGate != nullptr);
  assert(rhs->quantumGate != nullptr);

  // candidate block
  auto* cBlock = graph.acquireGateBlock();

  std::cerr << "Trying to fuse "
  << "lhs " << lhs->id << " and rhs " << rhs->id
  << " => candidate block " << cBlock->id << "\n";

  // set up connections
  llvm::SmallVector<int> cQubits;
  for (const auto& lWire : lhs->wires) {
    auto qubit = lWire.qubit;
    cQubits.push_back(qubit);

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
      cQubits.push_back(qubit);
      cBlock->wires.push_back(rWire);
    }
  }

  // check fusion eligibility: nqubits
  if (cBlock->nqubits() > config.maxNQubits) {
    std::cerr << CYAN("Rejected due to maxNQubits\n");
    graph.releaseGateBlock(cBlock);
    return nullptr;
  }

  // check fusion eligibility: opCount
  auto cGate = std::make_unique<QuantumGate>(
    rhs->quantumGate->lmatmul(*(lhs->quantumGate)));
  if (config.maxOpCount >= 0 &&
      cGate->opCount(config.zeroSkippingThreshold) > config.maxOpCount) {
    std::cerr << CYAN("Rejected due to OpCount\n");
    graph.releaseGateBlock(cBlock);
    return nullptr;
  }

  // accept candidate
  std::cerr << GREEN("Fusion accepted!\n");
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
    const CPUFusionConfig& config, CircuitGraph& graph, tile_iter_t tileIt) {
  int nFused = 0;
  auto tileNext = tileIt.next();
  if (tileNext == nullptr)
    return 0;

  for (unsigned q = 0; q < graph.nqubits; ++q) {
    auto* lBlock = (*tileIt)[q];
    auto* rBlock = (*tileNext)[q];
    auto* cBlock = computeCandidate(config, graph, lBlock, rBlock);
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
int traverseAndFuse(const CPUFusionConfig& config, CircuitGraph& graph) {
  int nFused = 0;
  auto it = graph.tile().begin();
  while (it != nullptr) {
    nFused += tryCrossWireFuse(config, graph, it);
    ++it;
  }
  return nFused;
}

} // anonymous namespace

void saot::applyCPUGateFusion(
    const CPUFusionConfig& config, CircuitGraph& graph) {
  int nFused = 0;
  do {
    nFused = traverseAndFuse(config, graph);
    graph.squeeze();
  } while (config.allowMultipleTraverse && nFused > 0);
}
