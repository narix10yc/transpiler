#include "simulation/ir_generator.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Pass.h"

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

using namespace saot;
using namespace llvm;
using namespace simulation;

void IRGenerator::createJitSession() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmParser();
    InitializeNativeTargetAsmPrinter();
    _jitter = std::move(cantFail(orc::LLJITBuilder().create()));
    cantFail(_jitter->addIRModule(orc::ThreadSafeModule(std::move(_module), std::move(_context))));
}

