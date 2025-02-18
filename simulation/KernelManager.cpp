#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"

#include "simulation/KernelManager.h"

#include <cast/Fusion.h>

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <ranges>

using namespace cast;
using namespace llvm;

void KernelManager::initJIT(
    int nThreads, OptimizationLevel optLevel, bool useLazyJIT, int verbose) {
  assert(nThreads > 0);
  assert(!isJITed() && "Already initialized");

  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();

  if (optLevel != OptimizationLevel::O0) {
    utils::TaskDispatcher dispatcher(nThreads);
    for (auto& [ctx, mod] :
        std::ranges::views::reverse(llvmContextModulePairs)) {
      // TODO: For some reason, MPM cannot be reused. For now we construct it
      // afresh for every module. Overhead is okay though.
      dispatcher.enqueue([&]() {
        // ChatGPT:
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
    if (verbose > 0)
      std::cout << "Applying LLVM Optimization....\n";
    dispatcher.sync(/* progressBar */ verbose > 0);
  }

  if (useLazyJIT) {
    // lazy JIT engine
  	orc::LLLazyJITBuilder jitBuilder;
  	jitBuilder.setNumCompileThreads(std::thread::hardware_concurrency());
  	auto lazyJIT = cantFail(jitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      cantFail(lazyJIT->addLazyIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
    }
    this->llvmJIT = std::move(lazyJIT);
    // eager compile all kernels
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  } else {
    // eager JIT engine
    orc::LLJITBuilder eagerJitBuilder;

    /// It seems not matter how many concurrency we set here.
    /// As long as we set it, we can invoke multiple lookup
  	eagerJitBuilder.setNumCompileThreads(10);
  	auto eagerJIT = cantFail(eagerJitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      cantFail(eagerJIT->addIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
    }
	  this->llvmJIT = std::move(eagerJIT);
    // eager compile all kernels
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  }
  this->llvmContextModulePairs.clear();
}

void KernelManager::ensureAllExecutable(int nThreads, bool progressBar) {
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

std::ostream& CPUKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CPU Kernel Gen Config ===\n")
     << "simd_s:    " << simd_s << "\n"
     << "precision:  " << precision << "\n"
     << "amp format: ";
  switch (this->ampFormat) {
    case AltFormat:
      os << "AltFormat\n"; break;
    case SepFormat:
      os << "SepFormat\n"; break;
    default:
      assert(0 && "Unreachable");
  }

  os << "useFMA     : " << useFMA << "\n"
     << "useFMS     : " << useFMS << "\n"
     << "usePDEP     : " << usePDEP << "\n"
     << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
    case UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case StackLoadMatElems:
      os << "StackLoadMatElems\n"; break;
    case StackLoadMatVecs:
      os << "StackLoadMatVecs\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}