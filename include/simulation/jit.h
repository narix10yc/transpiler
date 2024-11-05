#ifndef SIMULATION_JIT_H
#define SIMULATION_JIT_H

#include "simulation/ir_generator.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace saot::jit {

class JitEngine {
public:
    std::unique_ptr<llvm::orc::LLJIT> JIT;
    JitEngine(simulation::IRGenerator&);
};

} // namespace saot::jit

#endif // SIMULATION_JIT_H