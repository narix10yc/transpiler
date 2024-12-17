#ifndef SAOT_CIRCUITGRAPH_H
#define SAOT_CIRCUITGRAPH_H

#include "saot/QuantumGate.h"

#include <array>
#include <list>
#include <set>
#include <vector>

namespace saot {

class GateNode {
private:
  static int idCount;

public:
  struct Item {
    int qubit;
    GateNode* lhsGate;
    GateNode* rhsGate;
  };
  const int id;
  unsigned nqubits;
  GateMatrix gateMatrix;
  std::vector<Item> items;

  GateNode(const GateMatrix& gateMatrix, const std::vector<int>& qubits)
      : id(idCount++), nqubits(gateMatrix.nqubits()), gateMatrix(gateMatrix),
        items(gateMatrix.nqubits()) {
    assert(gateMatrix.nqubits() == qubits.size());
    for (unsigned i = 0; i < qubits.size(); i++)
      items[i] = {qubits[i], nullptr, nullptr};
  }

  Item* findQubit(int q) {
    for (auto& item : items) {
      if (item.qubit == q)
        return &item;
    }
    return nullptr;
  }

  const Item* findQubit(int q) const {
    for (const auto& item : items) {
      if (item.qubit == q)
        return &item;
    }
    return nullptr;
  }

  GateNode* findLHS(unsigned q) const {
    for (const auto& data : items) {
      if (data.qubit == q)
        return data.lhsGate;
    }
    return nullptr;
  }

  GateNode* findRHS(unsigned q) const {
    for (const auto& data : items) {
      if (data.qubit == q)
        return data.rhsGate;
    }
    return nullptr;
  }

  int connect(GateNode* rhsGate, int q = -1);

  std::vector<int> getQubits() const {
    std::vector<int> qubits(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
      qubits[i] = items[i].qubit;

    return qubits;
  }

  QuantumGate toQuantumGate() const {
    return QuantumGate(gateMatrix, getQubits());
  }
};

class GateBlock {
private:
  static int idCount;

public:
  struct Item {
    int qubit;
    GateNode* lhsEntry;
    GateNode* rhsEntry;
  };

  int id;
  std::vector<Item> items;
  std::unique_ptr<QuantumGate> quantumGate;

  GateBlock() : id(idCount++), items(), quantumGate(nullptr) {}

  GateBlock(GateNode* gateNode)
      : id(idCount++), items(),
        quantumGate(std::make_unique<QuantumGate>(gateNode->toQuantumGate())) {
    for (const auto& data : gateNode->items)
      items.push_back({data.qubit, gateNode, gateNode});
  }

  std::ostream& displayInfo(std::ostream& os) const;

  std::vector<GateNode*> getOrderedGates() const;

  size_t countGates() const { return getOrderedGates().size(); }

  int connect(GateBlock* rhsBlock, int q = -1);

  int nqubits() const { return items.size(); }

  Item* findQubit(int q) {
    for (auto& item : items) {
      if (item.qubit == q)
        return &item;
    }
    return nullptr;
  }

  const Item* findQubit(int q) const {
    for (const auto& item : items) {
      if (item.qubit == q)
        return &item;
    }
    return nullptr;
  }

  bool hasSameTargets(const GateBlock& other) const {
    if (nqubits() != other.nqubits())
      return false;
    for (const auto& data : other.items) {
      if (findQubit(data.qubit) == nullptr)
        return false;
    }
    return true;
  }

  // TODO: This should be identical to quantumGate->qubits
  // Find a way to remove the redundancy
  std::vector<int> getQubits() const {
    std::vector<int> vec(items.size());
    for (unsigned i = 0; i < items.size(); i++)
      vec[i] = items[i].qubit;
    return vec;
  }

  void internalFuse() { assert(false && "Not Implemented"); }
};

class CircuitGraph {
private:
  using row_t = std::array<GateBlock*, 36>;
  using tile_t = std::list<row_t>;
  using tile_iter_t = std::list<row_t>::iterator;
  using tile_riter_t = std::list<row_t>::reverse_iterator;
  using tile_const_iter_t = std::list<row_t>::const_iterator;
  tile_t _tile;

public:
  int nqubits;

  CircuitGraph() : _tile(1, {nullptr}), nqubits(0) {}

  static CircuitGraph QFTCircuit(int nqubits);
  static CircuitGraph ALACircuit(int nqubits, int nrounds);

  static CircuitGraph GetTestCircuit(
    const GateMatrix& gateMatrix, int nqubits, int nrounds);

  tile_t& tile() { return _tile; }
  const tile_t& tile() const { return _tile; }

  /// @brief Erase empty rows in the tile
  void eraseEmptyRows();

  bool isRowVacant(tile_iter_t it, const GateBlock* block) const {
    for (const auto& q : block->getQubits())
      if ((*it)[q] != nullptr)
        return false;
    return true;
  }

  tile_iter_t repositionBlockUpward(tile_iter_t it, size_t q_);
  tile_iter_t repositionBlockUpward(tile_riter_t it, size_t q_) {
    return repositionBlockUpward(--(it.base()), q_);
  }

  tile_riter_t repositionBlockDownward(tile_riter_t it, size_t q_);
  tile_riter_t repositionBlockDownward(tile_iter_t it, size_t q_) {
    return repositionBlockDownward(--std::make_reverse_iterator(it), q_);
  }

  void updateTileUpward();
  void updateTileDownward();

  /// @brief Try to insert block to a specified row. Three outcome may happen:
  /// - If \p it is vacant, insert \p block there. Otherwise,
  /// - If \p it+1 is vacant, insert \p block there. Otherwise,
  /// - Insert a separate row between \p it and \p it+1 and place \p block
  ///   there.
  tile_iter_t insertBlock(tile_iter_t it, GateBlock* block);

  void addGate(const QuantumGate& gate) {
    return addGate(gate.gateMatrix, gate.qubits);
  }

  void addGate(const GateMatrix& matrix, const std::vector<int>& qubits);

  /// @return ordered vector of blocks
  std::vector<GateBlock*> getAllBlocks() const;

  /// @brief Get the number of blocks with each size.
  /// @return ret[i] is the number of blocks with size i. Therefore, ret[0] is
  /// always 0, and ret.size() == largest_size + 1.
  std::vector<int> getBlockSizes() const;

  std::vector<std::vector<int>> getBlockOpCountHistogram() const;

  size_t countBlocks() const { return getAllBlocks().size(); }

  size_t countGates() const {
    const auto allBlocks = getAllBlocks();
    size_t sum = 0;
    for (const auto& block : allBlocks)
      sum += block->countGates();
    return sum;
  }

  size_t countTotalOps() const {
    const auto allBlocks = getAllBlocks();
    size_t sum = 0;
    for (const auto& block : allBlocks) {
      assert(block->quantumGate != nullptr);
      sum += block->quantumGate->opCount();
    }
    return sum;
  }

  void relabelBlocks() const;

  /// @brief Console print the tile.
  /// @param verbose If > 1, also print the address of each row in front
  std::ostream& print(std::ostream& os = std::cerr, int verbose = 1) const;

  std::ostream& displayInfo(
    std::ostream& os = std::cerr, int verbose = 1) const;
};

}; // namespace saot

#endif // SAOT_CIRCUITGRAPH_H