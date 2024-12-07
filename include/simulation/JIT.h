#ifndef SIMULATION_JIT_H
#define SIMULATION_JIT_H

#include "llvm/IR/Module.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace saot {

std::unique_ptr<llvm::orc::LLJIT>
createJITSession(std::unique_ptr<llvm::Module> = nullptr);

} // namespace saot

#endif // SIMULATION_JIT_H