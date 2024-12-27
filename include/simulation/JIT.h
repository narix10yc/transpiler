#ifndef SIMULATION_JIT_H
#define SIMULATION_JIT_H

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"

#include "llvm/Passes/OptimizationLevel.h"

namespace saot {

std::unique_ptr<llvm::orc::LLJIT> createJITSession(
  std::unique_ptr<llvm::Module>, std::unique_ptr<llvm::LLVMContext>);

inline std::unique_ptr<llvm::orc::LLJIT> createJITSession() {
  return createJITSession(nullptr, nullptr);
}

void applyLLVMOptimization(
    llvm::Module& llvmModule, const llvm::OptimizationLevel& optLevel);

} // namespace saot

#endif // SIMULATION_JIT_H