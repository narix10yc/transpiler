#ifndef SIMULATION_JIT_H
#define SIMULATION_JIT_H

#include "simulation/ir_generator.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace saot::jit {

class JitEngine {
    std::unique_ptr<llvm::orc::LLJIT> JIT;
public:
    JitEngine(simulation::IRGenerator&);

    void dumpNativeAssembly(llvm::raw_ostream&);
};

} // namespace saot::jit

#endif // SIMULATION_JIT_H