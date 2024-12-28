#ifndef SAOT_CIRCUITGRAPH_H
#define SAOT_CIRCUITGRAPH_H

#include "saot/QuantumGate.h"
#include "saot/CircuitGraphContext.h"
#include "utils/List.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <vector>

namespace saot {

// We forward CircuitGraph declaration here because GateNode constructor needs
// it to set up connection
class CircuitGraph;

class GateNode {
public:
  struct ConnectionInfo {
    int qubit;
    GateNode* lhsGate;
    GateNode* rhsGate;
  };

  int id;
  QuantumGate* quantumGate;
  std::vector<ConnectionInfo> connections;

  GateNode(QuantumGate* quantumGate, const CircuitGraph& graph);

  ConnectionInfo* findConnection(int qubit) {
    for (auto& item : connections) {
      if (item.qubit == qubit)
        return &item;
    }
    return nullptr;
  }

  const ConnectionInfo* findConnection(int qubit) const {
    for (const auto& item : connections) {
      if (item.qubit == qubit)
        return &item;
    }
    return nullptr;
  }

  void connect(GateNode* rhsGate, int q);
};

/// \brief A \c GateBlock is an aggregate of \c GateNode 's
/// It is the primary layer for gate fusion. Aggregation relation is maintained
/// by the \c wires variable, which is internally a vector of \c WireInfo that
/// specifies qubit, left entry node, and right entry node.
class GateBlock {
public:
  struct WireInfo {
    int qubit;
    GateNode* lhsEntry;
    GateNode* rhsEntry;
  };

  int id;
  std::vector<WireInfo> wires;
  QuantumGate* quantumGate;

  GateBlock();
  explicit GateBlock(GateNode* gateNode);

  std::ostream& displayInfo(std::ostream& os) const;

  std::vector<GateNode*> getOrderedGates() const;

  size_t countGates() const { return getOrderedGates().size(); }

  int nqubits() const { return wires.size(); }

  WireInfo* findWire(int qubit) {
    for (auto& wire : wires) {
      if (wire.qubit == qubit)
        return &wire;
    }
    return nullptr;
  }

  const WireInfo* findWire(int qubit) const {
    for (const auto& wire : wires) {
      if (wire.qubit == qubit)
        return &wire;
    }
    return nullptr;
  }

  bool hasSameTargets(const GateBlock& other) const {
    if (nqubits() != other.nqubits())
      return false;
    for (const auto& data : other.wires) {
      if (findWire(data.qubit) == nullptr)
        return false;
    }
    return true;
  }

  void internalFuse() { assert(false && "Not Implemented"); }
};


class CircuitGraph {
private:
  CircuitGraphContext _context;
public:
  using row_t = std::array<GateBlock*, 36>;
  using tile_t = utils::List<row_t>;
  using list_node_t = tile_t::Node;
  // iterator
  using iter_t = tile_t::iterator;
  // using citer_t = tile_t::const_iterator;
  // using riter_t = tile_t::reverse_iterator;
  // using criter_t = tile_t::const_reverse_iterator;

private:
  tile_t _tile;
public:
  static int GateNodeCount;
  static int GateBlockCount;

  int nqubits;

  CircuitGraph() : _tile(), nqubits(0) {
    _tile.emplace_back();
  }

  static CircuitGraph QFTCircuit(int nqubits);
  static CircuitGraph ALACircuit(int nqubits, int nrounds);

  static CircuitGraph GetTestCircuit(
    const GateMatrix& gateMatrix, int nqubits, int nrounds);

  tile_t& tile() { return _tile; }
  const tile_t& tile() const { return _tile; }

  CircuitGraphContext& getContext() { return _context; }

  template<typename... Args>
  QuantumGate* acquireQuantumGate(Args&&... args) {
    return _context.quantumGatePool.acquire(std::forward<Args>(args)...);
  }

  template<typename... Args>
  GateNode* acquireGateNode(Args&&... args) {
    return _context.gateNodePool.acquire(std::forward<Args>(args)...);
  }

  template<typename... Args>
  GateBlock* acquireGateBlock(Args&&... args) {
    return _context.gateBlockPool.acquire(std::forward<Args>(args)...);
  }

  void releaseQuantumGate(QuantumGate* gate) {
    _context.quantumGatePool.release(gate);
  }

  void releaseGateNode(GateNode* gateNode) {
    _context.gateNodePool.release(gateNode);
  }

  void releaseGateBlock(GateBlock* gateBlock) {
    _context.gateBlockPool.release(gateBlock);
  }

  bool isManaging(const QuantumGate* gate) const {
    return _context.quantumGatePool.isInPool(gate);
  }

  bool isManaging(const GateNode* gateNode) const {
    return _context.gateNodePool.isInPool(gateNode);
  }

  bool isManaging(const GateBlock* gateBlock) const {
    return _context.gateBlockPool.isInPool(gateBlock);
  }

  /// Append a quantum gate to the tile. Quantum gate must be managed by
  /// \c *this
  void appendGate(QuantumGate* quantumGate);

  /// @brief Erase empty rows in the tile
  void eraseEmptyRows();

  bool isRowVacant(const row_t& row, const GateBlock* block) const {
    for (const auto& q : block->quantumGate->qubits)
      if (row[q] != nullptr)
        return false;
    return true;
  }

  /// Update connections of blocks. By assumption nodes connections are always
  /// preserved. In Debug mode this function will check for node connections.
  void updateBlockConnections(iter_t it, int q);

  list_node_t* repositionBlockUpward(list_node_t* ln, int q);
  iter_t repositionBlockUpward(iter_t it, int q) {
    return iter_t(repositionBlockUpward(it.raw_ptr(), q));
  }

  list_node_t* repositionBlockDownward(list_node_t* ln, int q);
  iter_t repositionBlockDownward(iter_t it, int q) {
    return iter_t(repositionBlockDownward(it.raw_ptr(), q));
  }

  // Squeeze graph and make it compact
  void squeeze();

  /// @brief Try to insert block to a specified row. Three outcome may happen:
  /// - If \p it is vacant, insert \p block there. Otherwise,
  /// - If \p it+1 is vacant, insert \p block there. Otherwise,
  /// - Insert a separate row between \p it and \p it+1 and place \p block
  ///   there.
  iter_t insertBlock(iter_t it, GateBlock* block);

  /// @brief Get the number of blocks with each size.
  /// @return ret[i] is the number of blocks with size i. Therefore, ret[0] is
  /// always 0, and ret.size() == largest_size + 1.
  std::vector<int> getBlockSizes() const;

  /// @return ordered vector of blocks
  std::vector<GateBlock*> getAllBlocks() const;

  // std::vector<std::vector<int>> getBlockOpCountHistogram() const;
  //
  // size_t countBlocks() const { return getAllBlocks().size(); }
  //
  // size_t countGates() const {
  //   const auto allBlocks = getAllBlocks();
  //   size_t sum = 0;
  //   for (const auto& block : allBlocks)
  //     sum += block->countGates();
  //   return sum;
  // }
  //
  // size_t countTotalOps() const {
  //   const auto allBlocks = getAllBlocks();
  //   size_t sum = 0;
  //   for (const auto& block : allBlocks) {
  //     assert(block->quantumGate != nullptr);
  //     sum += block->quantumGate->opCount();
  //   }
  //   return sum;
  // }
  //
  // void relabelBlocks() const;

  /// @brief Console print the tile.
  /// @param verbose If > 1, also print the address of each row in front
  std::ostream& print(std::ostream& os = std::cerr, int verbose = 1) const;

  std::ostream& displayInfo(
    std::ostream& os = std::cerr, int verbose = 1) const;
};

}; // namespace saot

#endif // SAOT_CIRCUITGRAPH_H