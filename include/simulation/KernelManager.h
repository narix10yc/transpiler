#ifndef SIMULATION_KERNELMANAGER_H
#define SIMULATION_KERNELMANAGER_H

#define CPU_KERNEL_TYPE void(void*, uint64_t, uint64_t, const void*)

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Passes/OptimizationLevel.h>

#include "saot/QuantumGate.h"

namespace saot {

class CircuitGraph;

struct KernelInfo {
  enum KernelType {
    CPU_Gate, CPU_Measure, GPU_Gate, GPU_Measure
  };
  std::function<CPU_KERNEL_TYPE> executable;

  KernelType type;
  int precision;
  std::string llvmFuncName;
  QuantumGate gate;
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

class KernelManager {
  struct ContextModulePair {
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;
  };

  std::vector<ContextModulePair> llvmContextModulePairs;

  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;

  std::vector<KernelInfo> _kernels;

  std::mutex mtx;

  /// A thread-safe version that creates a new llvm Module
  ContextModulePair& createNewLLVMModule(const std::string& name) {
    assert(!isJITed());
    std::lock_guard<std::mutex> lock(mtx);
    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvmContextModulePairs.emplace_back(
      std::move(ctx), std::make_unique<llvm::Module>(name, *ctx));
    return llvmContextModulePairs.back();
  }
public:
  KernelManager()
    : llvmContextModulePairs()
    , llvmJIT(nullptr)
    , _kernels() {}

  const std::vector<KernelInfo>& kernels() const { return _kernels; }
  std::vector<KernelInfo>& kernels() { return _kernels; }

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
      bool useLazyJIT = false);

  bool isJITed() const {
    assert(llvmJIT == nullptr ||
           (llvmContextModulePairs.empty() && llvmJIT != nullptr));
    return llvmJIT != nullptr;
  }

  /// A function that takes in 4 arguments (void*, uint64_t, uint64_t,
  /// void*) and returns void. Arguments are: pointer to statevector array,
  /// taskID begin, taskID end, and pointer to matrix array (could be null).
  KernelManager& genCPUKernel(
      const CPUKernelGenConfig& config,
      const QuantumGate& gate, const std::string& funcName);

  /// A function that takes in 4 arguments (void*, uint64_t, uint64_t,
  /// void*) and returns void. Arguments are: pointer to statevector array,
  /// taskID begin, taskID end, and pointer to measurement probability to write on
  KernelManager& genCPUMeasure(
      const CPUKernelGenConfig& config, int q, const std::string& funcName);

  KernelManager& genCPUFromGraph(
      const CPUKernelGenConfig& config,
      const CircuitGraph& graph, const std::string& graphName);

  std::vector<KernelInfo*> collectCPUGraphKernels(const std::string& graphName);

  void ensureExecutable(KernelInfo& kernel) {
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
    std::cerr << "Kernel " << kernel.llvmFuncName << " addr " << (void*)addr << "\n";
    {
      std::lock_guard<std::mutex> lock(mtx);
      kernel.executable = addr;
    }
  }

  void ensureAllExecutable(int nThreads = 1);

  void applyCPUKernel(
      void* sv, int nQubits, KernelInfo& kernelInfo);

  void applyCPUKernel(void* sv, int nQubits, const std::string& funcName);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, KernelInfo& kernelInfo, int nThreads);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, const std::string& funcName, int nThreads);
};


} // namespace saot

#endif // SIMULATION_KERNELMANAGER_H
