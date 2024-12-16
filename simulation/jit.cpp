#include "simulation/ir_generator.h"
#include "simulation/JIT.h"

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/Support/TargetSelect.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

using namespace saot;
using namespace llvm;
using namespace simulation;

void IRGenerator::createJitSession() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();
  _jitter = std::move(cantFail(orc::LLJITBuilder().create()));
  cantFail(_jitter->addIRModule(
      orc::ThreadSafeModule(std::move(_module), std::move(_context))));
}

std::unique_ptr<orc::LLJIT> saot::createJITSession(
    std::unique_ptr<Module> llvmModule,
    std::unique_ptr<LLVMContext> llvmContext) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();
  auto jit = std::move(cantFail(orc::LLJITBuilder().create()));
  if (llvmModule && llvmContext) {
    cantFail(jit->addIRModule(
      orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext))));
  }
  return jit;
}