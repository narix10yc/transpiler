#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Passes/PassBuilder.h"

#include "simulation/KernelManager.h"

#include "cast/Fusion.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <ranges>

using namespace cast;
using namespace llvm;

KernelManagerBase::ContextModulePair&
KernelManagerBase::createNewLLVMModule(const std::string& name) {
  std::lock_guard<std::mutex> lock(mtx);
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvmContextModulePairs.emplace_back(
    std::move(ctx), std::make_unique<llvm::Module>(name, *ctx));
  return llvmContextModulePairs.back();
}

void KernelManagerBase::applyLLVMOptimization(
    int nThreads, OptimizationLevel optLevel, bool progressBar) {
  assert(nThreads > 0);
  if (optLevel == OptimizationLevel::O0)
    return;

  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& [ctx, mod] : llvmContextModulePairs) {
    // TODO: For some reason, MPM cannot be reused. For now we construct it
    // afresh for every module. Overhead is okay though.
    dispatcher.enqueue([&]() {
      // These must be declared in this order so that they are destroyed in
      // the correct order due to inter-analysis-manager references.
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      PassBuilder PB;

      PB.registerLoopAnalyses(LAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerModuleAnalyses(MAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
      ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(optLevel);

      MPM.run(*mod, MAM);
    });
  }
  if (progressBar)
    std::cout << "Applying LLVM Optimization....\n";
  dispatcher.sync(progressBar);
}

void CPUKernelManager::ensureAllExecutable(int nThreads, bool progressBar) {
  assert(nThreads > 0);
  if (nThreads == 1) {
    for (auto& kernel : _kernels)
      ensureExecutable(kernel);
    return;
  }

  // multi-thread compile
  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& kernel : _kernels) {
	  dispatcher.enqueue([this, &kernel]() {
      ensureExecutable(kernel);
	  });
  }
  if (progressBar)
    std::cout << "Ensure All Executables...\n";
  dispatcher.sync(progressBar);
}
