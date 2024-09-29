#ifndef SIMULATION_SAOTCONTEXT_H
#define SIMULATION_SAOTCONTEXT_H

#include "simulation/ir_generator.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace saot::jit {

class JitEngine {
    std::unique_ptr<llvm::orc::LLJIT> JIT;
public:
    JitEngine(std::unique_ptr<simulation::IRGenerator>);

    void dumpNativeAssembly(llvm::raw_ostream&);
};

} // namespace saot::jit

#endif // SIMULATION_SAOTCONTEXT_H