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

#define DEBUG_TYPE "kernel-mgr-cuda"
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

  // Check registry info to make sure LLVM is built for NVPTX
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    errs() << RED("[Error]: ") << "Failed to lookup target. Error trace: "
           << err << "\n";
    return;
  }

  // Query device for compute capability
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);

  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

  std::ostringstream archOss;
  archOss << "sm_" << major << minor;
  std::string archString = archOss.str();

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
      targetTriple,   // target triple
      archString,     // cpu
      "",             // features
      {},             // options
      std::nullopt    // RM
    );
  };

  for (auto& [ctx, mod] : llvmContextModulePairs) {
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(createTargetMachine()->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed!\n";
      return;
    }
  }

  assert(_cudaKernels.size() == llvmContextModulePairs.size());
  utils::TaskDispatcher dispatcher(nThreads);

  for (unsigned i = 0; i < _cudaKernels.size(); i++) {
    dispatcher.enqueue([&, i=i]() {
      raw_svector_ostream sstream(_cudaKernels[i].ptxString);
      legacy::PassManager passManager;
      if (createTargetMachine()->addPassesToEmitFile(
          passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
        errs() << "The target machine can't emit a file of this type\n";
        return;
      }
      passManager.run(*llvmContextModulePairs[i].llvmModule);
    });
  }
  if (verbose > 0)
    std::cout << "Emitting PTX codes...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  jitState = JIT_PTXEmitted;
}

#ifdef CAST_USE_CUDA
void CUDAKernelManager::initCUJIT(int nThreads, int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nThreads > 0);
  auto nKernels = _cudaKernels.size();
  if (nKernels < nThreads) {
    std::cerr << YELLOW("[Warning] ")
      << "Calling initCUJIT with "
      << nThreads << " threads when there are only "
      << nKernels << " kernels. Set nThreads to "
      << nKernels << " instead.\n";
    nThreads = nKernels;
  }

  cuInit(0);
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, 0);

  utils::TaskDispatcher dispatcher(nThreads);

  // Create CUDA contexts. Each thread creates and manages its own CUDA context.
  cuContexts.resize(nThreads);
  for (unsigned t = 0; t < nThreads; ++t) {
    dispatcher.enqueue([&]() {
      auto workerID = dispatcher.getWorkerID();
      CUcontext* cuContextPtr = &cuContexts[workerID];
      CUresult cuResult = cuCtxCreate(cuContextPtr, 0, cuDevice);
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to create CUDA context. Error code " 
                  << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " created CUcontext " << *cuContextPtr
                    << " at " << cuContextPtr << "\n";
        );
      }
    });
  }
  dispatcher.sync();

  // Load PTX codes
  cuTuples.resize(nKernels);
  assert(nKernels == llvmContextModulePairs.size());
  
  // TODO: Currently ptxString is captured by value. This seems to be due to the
  // property of llvm::SmallVector<char, 0> -- calling str() returns an empty
  // StringRef.
  // One fix is to replace PTXStringType from SmallVector<char, 0> to 
  // std::string. Then we need to adjust emitPTX accordingly.
  for (unsigned i = 0; i < nKernels; ++i) {
    std::string ptxString(_cudaKernels[i].ptxString.str());
    CUcontext* cuContextPtr = &(cuTuples[i].cuContext);
    CUmodule* cuModulePtr = &(cuTuples[i].cuModule);
    CUfunction* cuFunctionPtr = &(cuTuples[i].cuFunction);
    const char* funcName = _cudaKernels[i].llvmFuncName.c_str();
    dispatcher.enqueue([=, this, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();
      CUresult cuResult;

      cuResult = cuCtxSetCurrent(cuContexts[workerID]);
      *cuContextPtr = cuContexts[workerID];
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to set CUDA context. Error code " 
                  << cuResult << "\n";
        return;
      }
      cuResult = cuModuleLoadData(cuModulePtr, ptxString.c_str());
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                << " failed to create CUDA module " << i << ". Error code "
                << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " created CUmodule " << *cuModulePtr
                    << " at " << cuModulePtr << "\n";
        );
      }

      cuResult = cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName);
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to load CUDA function " << i << ". Error code "
                  << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " loaded CUfunction " << *cuFunctionPtr
                    << " at " << cuFunctionPtr << "\n";
        );
      }
    });
  }
  if (verbose > 0)
    std::cout << "Loading PTX codes and getting CUDA functions...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  jitState = JIT_CUFunctionLoaded;
}

void CUDAKernelManager::launchCUDAKernel(
    void* dData, int nQubits, int kernelIdx) {
  auto cuTuple = cuTuples[kernelIdx];
  cuCtxSetCurrent(cuTuple.cuContext);

  // This corresponds to 128 threads per block
  int blockSizeInNumBits = 5;
  int nGateQubits = _cudaKernels[kernelIdx].gate->nQubits();
  int gridSizeInNumBits = nQubits - blockSizeInNumBits - nGateQubits;
  assert(gridSizeInNumBits >= 0 && "gridSize must be positive");

  unsigned blockSize = 1 << blockSizeInNumBits;
  unsigned gridSize = 1 << (nQubits - blockSizeInNumBits - nGateQubits);

  void* cMatPtr = 
    _cudaKernels[kernelIdx].gate->gateMatrix.getConstantMatrix()->data();
  CUdeviceptr dArrDevicePtr = (CUdeviceptr)dData;
  void* kernelParams[] = { &dData, &cMatPtr };

  // std::cerr << "Launching kernel <<<" << gridSize << ", " << blockSize << ">>>" << "\n";
  // std::cerr << "Kernel parameter " << (kernelParams[0])
  //           << ", dArrDevicePtr = " << (void*)dArrDevicePtr
  //           << ", &dArrDevicePtr = " << &dArrDevicePtr << "\n";

  auto cuResult = cuLaunchKernel(
    cuTuple.cuFunction, 
    gridSize, 1, 1,  // grid dim
    blockSize, 1, 1, // block dim
    0,              // shared mem size
    0,              // stream
    kernelParams,  // kernel params
    nullptr         // extra options
  );
  if (cuResult != CUDA_SUCCESS) {
    std::cerr << RED("[CUDA Driver Error]:") << "Failed to launch kernel"
              << "<<<" << gridSize << ", " << blockSize << ">>>. Error code: "
              << cuResult << "\n"; 
  }
}


#endif // CAST_USE_CUDA

#undef DEBUG_TYPE