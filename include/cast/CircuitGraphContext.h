#ifndef CAST_CIRCUITGRAPHCONTEXT_H
#define CAST_CIRCUITGRAPHCONTEXT_H

#include "utils/ObjectPool.h"

namespace cast {

class GateNode;
class GateBlock;

class CircuitGraphContext {
public:
  static int GateNodeCount;
  static int GateBlockCount;

  /// Memory management
  utils::ObjectPool<GateNode> gateNodePool;
  utils::ObjectPool<GateBlock> gateBlockPool;

  CircuitGraphContext() = default;
};

} // namespace cast

#endif // CAST_CIRCUITGRAPHCONTEXT_H
