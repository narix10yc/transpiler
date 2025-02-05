#ifndef SAOT_CIRCUITGRAPHCONTEXT_H
#define SAOT_CIRCUITGRAPHCONTEXT_H

#include "utils/ObjectPool.h"

namespace saot {

class GateNode;
class GateBlock;

class CircuitGraphContext {
public:
  /// Memory management
  utils::ObjectPool<GateNode> gateNodePool;
  utils::ObjectPool<GateBlock> gateBlockPool;

  CircuitGraphContext() = default;
};

} // namespace saot

#endif // SAOT_CIRCUITGRAPHCONTEXT_H
