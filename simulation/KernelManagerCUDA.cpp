
#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/Verifier.h"

#include "cast/QuantumGate.h"
#include "cast/CircuitGraph.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include "utils/Formats.h"

#define DEBUG_TYPE "codegen-cuda"
#include <llvm/Support/Debug.h>
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

std::ostream& CUDAKernelGenConfig::displayInfo(std::ostream& os) const {
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

void CUDAKernelManager::emitPTX(
    int nThreads, OptimizationLevel optLevel, int verbose) {
  assert(nThreads > 0);

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

  for (auto& [ctx, mod] : llvmContextModulePairs) {
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(targetMachine->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed!\n";
      return;
    }
  }

  assert(_kernels.size() == llvmContextModulePairs.size());
  utils::TaskDispatcher dispatcher(nThreads);

  for (unsigned i = 0; i < _kernels.size(); i++) {
    dispatcher.enqueue([&, i=i, tm=targetMachine]() {
      raw_svector_ostream sstream(_kernels[i].ptxString);
      legacy::PassManager passManager;
      if (tm->addPassesToEmitFile(
          passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
        errs() << "The target machine can't emit a file of this type\n";
        return;
      }
      passManager.run(*llvmContextModulePairs[i].llvmModule);
    });
  }
  dispatcher.sync();
}

#undef DEBUG_TYPE