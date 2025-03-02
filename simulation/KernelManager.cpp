#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/Verifier.h"

#include "simulation/KernelManager.h"

#include "cast/Fusion.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <ranges>

using namespace cast;
using namespace llvm;

void KernelManager::applyLLVMOptimization(
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

void KernelManager::initJIT(
    int nThreads, OptimizationLevel optLevel, bool useLazyJIT, int verbose) {
  assert(nThreads > 0);
  assert(!isJITed() && "Already initialized");

  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  if (useLazyJIT) {
    // lazy JIT engine
  	orc::LLLazyJITBuilder jitBuilder;
    /// It seems not matter how many concurrency we set here.
    /// As long as we set it, we can invoke multiple lookup, and we can 
    /// control the actual number of threads via our custom TaskDispatcher
  	jitBuilder.setNumCompileThreads(nThreads);
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
  	eagerJitBuilder.setNumCompileThreads(nThreads);
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


void KernelManager::initJITForPTXEmission(
    int nThreads, OptimizationLevel optLevel, int verbose) {
  assert(nThreads > 0);
  assert(!isJITed() && "Already initialized");

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  // std::string targetTriple = sys::getDefaultTargetTriple();
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    errs() << RED("[Error]: ") << err << "\n";
    return;
  }
  errs() << "Target " << target->getName() << "\n";

  std::string cpu = "sm_70";
  // std::string cpu = "generic";
  auto* targetMachine = target->createTargetMachine(
    targetTriple,   // target triple
    cpu,            // cpu
    "",             // features
    {},             // options
    std::nullopt    // RM
  );

  llvm::SmallString<8U> data_ptx;
  llvm::raw_svector_ostream dest_ptx(data_ptx);

  legacy::PassManager passManager;
  if (targetMachine->addPassesToEmitFile(
      passManager, dest_ptx, nullptr, CodeGenFileType::AssemblyFile)) {
    errs() << "The target machine can't emit a file of this type\n";
    return;
  }
  for (auto& [ctx, mod] : llvmContextModulePairs) {
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(targetMachine->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed!\n";
      return;
    }
    passManager.run(*mod);
  }

  auto ptxCode = dest_ptx.str();
  std::cerr.write(ptxCode.begin(), ptxCode.size());
  std::cerr << "\n";
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

std::ostream& GPUKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== GPU Kernel Gen Config ===\n")
     << "precision:  " << precision << "\n";

  os << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
    case UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case LoadInDefaultMemSpace:
      os << "LoadInDefaultMemSpace\n"; break;
    case LoadInConstMemSpace:
      os << "LoadInConstMemSpace\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}