#ifndef SIMULATION_KERNELMANAGER_H
#define SIMULATION_KERNELMANAGER_H

#define CPU_KERNEL_TYPE void(void*, uint64_t, uint64_t, const void*)

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Passes/OptimizationLevel.h>

#include "cast/QuantumGate.h"
#include <memory>

#ifdef CAST_USE_CUDA
  #include <cuda.h>
  // TODO: We may not need cuda_runtime (also no need to link CUDA::cudart)
  #include <cuda_runtime.h>
#endif // CAST_USE_CUDA

namespace cast {

  namespace internal {
    /// mangled name is formed by 'G' + <length of graphName> + graphName
    /// @return mangled name
    std::string mangleGraphName(const std::string& graphName);

    std::string demangleGraphName(const std::string& mangledName);
  } // namespace internal

class CircuitGraph;

class KernelManagerBase {
protected:
  struct ContextModulePair {
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;
  };
  // A vector of pairs of LLVM context and module. Expected to be cleared after
  // calling \c initJIT
  std::vector<ContextModulePair> llvmContextModulePairs;
  std::mutex mtx;

  /// A thread-safe version that creates a new llvm Module
  ContextModulePair& createNewLLVMContextModulePair(const std::string& name);

  /// Apply LLVM optimization to all modules inside \c llvmContextModulePairs
  /// As a private member function, this function will be called by \c initJIT
  /// and \c initJITForPTXEmission
  void applyLLVMOptimization(
      int nThreads, llvm::OptimizationLevel optLevel, bool progressBar);
};

/*
  CPU
*/
struct CPUKernelInfo {
  enum KernelType {
    CPU_Gate, CPU_Measure
  };
  std::function<CPU_KERNEL_TYPE> executable;

  KernelType type;
  int precision;
  std::string llvmFuncName;
  std::shared_ptr<QuantumGate> gate;
  // extra information
  int simd_s;
  int opCount;
  int nLoBits;
};

struct CPUKernelGenConfig {
  enum AmpFormat { AltFormat, SepFormat };
  enum MatrixLoadMode { UseMatImmValues, StackLoadMatElems, StackLoadMatVecs };

  int simd_s = 2;
  int precision = 64;
  AmpFormat ampFormat = AltFormat;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposit from BMI2
  bool usePDEP = false;
  bool forceDenseKernel = false;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  MatrixLoadMode matrixLoadMode = UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;

  // TODO: set up default configurations
  static const CPUKernelGenConfig NativeDefaultF32;
  static const CPUKernelGenConfig NativeDefaultF64;
};

class CPUKernelManager : public KernelManagerBase {
  std::vector<CPUKernelInfo> _kernels;
  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;
public:
  CPUKernelManager()
    : KernelManagerBase()
    , _kernels()
    , llvmJIT(nullptr) {}

  const std::vector<CPUKernelInfo>& kernels() const { return _kernels; }
  std::vector<CPUKernelInfo>& kernels() { return _kernels; }

  /// Initialize JIT session. When succeeds, \c llvmContextModulePairs
  /// will be cleared and \c llvmJIT will be non-null. This function can only be
  /// called once and cannot be undone.
  /// \param nThreads number of threads to use.
  /// \param optLevel Apply LLVM optimization passes.
  /// \param useLazyJIT If true, use lazy compilation features provided by LLVM
  /// ORC JIT engine. This means all kernels only get compiled just before being
  /// called. If set to false, all kernels are ready to be executed when this
  /// function returns (good for benchmarks).
  void initJIT(
      int nThreads = 1,
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
      bool useLazyJIT = false,
      int verbose = 0);

  bool isJITed() const {
    assert(llvmJIT == nullptr ||
           (llvmContextModulePairs.empty() && llvmJIT != nullptr));
    return llvmJIT != nullptr;
  }

  /// A function that takes in 4 arguments (void*, uint64_t, uint64_t,
  /// void*) and returns void. Arguments are: pointer to statevector array,
  /// taskID begin, taskID end, and pointer to matrix array (could be null).
  CPUKernelManager& genCPUGate(
      const CPUKernelGenConfig& config,
      std::shared_ptr<QuantumGate> gate, const std::string& funcName);

  /// A function that takes in 4 arguments (void*, uint64_t, uint64_t,
  /// void*) and returns void. Arguments are: pointer to statevector array,
  /// taskID begin, taskID end, and pointer to measurement probability to write on
  CPUKernelManager& genCPUMeasure(
      const CPUKernelGenConfig& config, int q, const std::string& funcName);

  CPUKernelManager& genCPUGatesFromCircuitGraph(
      const CPUKernelGenConfig& config,
      const CircuitGraph& graph, const std::string& graphName);

  std::vector<CPUKernelInfo*>
  collectCPUKernelsFromCircuitGraph(const std::string& graphName);

  void ensureExecutable(CPUKernelInfo& kernel) {
    // Note: We do not actually need the lock here
    // as it is expected (at least now) each KernelInfo is accesses by a unique
    // thread
    // TODO: we could inline this function into \c initJIT. Maybe introduce a
    // lock inside \c initJIT
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (kernel.executable)
        return;
    }
    auto addr = cantFail(llvmJIT->lookup(kernel.llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
    // std::cerr << "Kernel " << kernel.llvmFuncName << " addr " << (void*)addr << "\n";
    {
      std::lock_guard<std::mutex> lock(mtx);
      kernel.executable = addr;
    }
  }

	void dumpIR(const std::string &funcName, llvm::raw_ostream &os);

  void ensureAllExecutable(int nThreads = 1, bool progressBar = false);

  void applyCPUKernel(
      void* sv, int nQubits, CPUKernelInfo& kernelInfo);

  void applyCPUKernel(void* sv, int nQubits, const std::string& funcName);

	void applyCPUKernel(void* sv, int nQubits, const std::string& funcName, const void* pMatArg);

	void applyCPUKernel(void* sv, int nQubits, CPUKernelInfo& kernel, const void* pMatArg);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, CPUKernelInfo& kernelInfo, int nThreads);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, const std::string& funcName, int nThreads);
};

/*
  CUDA
*/
struct CUDAKernelInfo {
  enum KernelType {
    CUDA_Gate, CUDA_Measure
  };
  // We expect large stream writes anyway, so always trigger heap allocation.
  using PTXStringType = llvm::SmallString<0>;
  PTXStringType ptxString;
  KernelType type;
  int precision;
  std::string llvmFuncName;
  std::shared_ptr<QuantumGate> gate;
  // extra information
  int opCount;
};

struct CUDAKernelGenConfig {
  enum MatrixLoadMode { 
    UseMatImmValues, LoadInDefaultMemSpace, LoadInConstMemSpace
  };
  int precision = 64;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  bool forceDenseKernel = false;
  MatrixLoadMode matrixLoadMode = UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  std::vector<CUDAKernelInfo> _cudaKernels;

  enum JITState { JIT_Uninited, JIT_PTXEmitted, JIT_CUFunctionLoaded };
  JITState jitState;
public:
  CUDAKernelManager()
    : KernelManagerBase()
    , _cudaKernels()
    , jitState(JIT_Uninited) {}

  std::vector<CUDAKernelInfo>& kernels() { return _cudaKernels; }
  const std::vector<CUDAKernelInfo>& kernels() const { return _cudaKernels; }

  CUDAKernelManager& genCUDAGate(
      const CUDAKernelGenConfig& config,
      std::shared_ptr<QuantumGate> gate, const std::string& funcName);

  CUDAKernelManager& genCUDAGatesFromCircuitGraph(
      const CUDAKernelGenConfig& config,
      const CircuitGraph& graph, const std::string& graphName);
    
  void emitPTX(
      int nThreads = 1,
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
      int verbose = 0);

  std::string getPTXString(int idx) const {
    return std::string(_cudaKernels[idx].ptxString.str());
  }

#ifdef CAST_USE_CUDA
private:
  /// Every CUDA module will contain exactly one CUDA function. Multiple CUDA
  /// modules may share the same CUDA context. For multi-threading JIT session,
  /// we create \c nThread number of CUDA contexts.
  struct CUDATuple {
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
  };
  // Every thread will manage its own CUcontext.
  std::vector<CUcontext> cuContexts;
  std::vector<CUDATuple> cuTuples;
public:
  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() {
    for (auto& [ctx, mod, func] : cuTuples)
      cuModuleUnload(mod);
    for (auto& ctx : cuContexts)
      cuCtxDestroy(ctx);
  }

  /// @brief Initialize CUDA JIT session by loading PTX strings into CUDA
  /// context and module. This function can only be called once and cannot be
  /// undone. This function assumes \c emitPTX is already called.
  void initCUJIT(int nThreads = 1, int verbose = 0);

  ///
  void launchCUDAKernel(void* dData, int nQubits, int kernelIdx);

#endif // CAST_USE_CUDA
};


} // namespace cast

#endif // SIMULATION_KERNELMANAGER_H
