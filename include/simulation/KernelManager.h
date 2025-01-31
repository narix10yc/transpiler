#ifndef SIMULATION_KERNELMANAGER_H
#define SIMULATION_KERNELMANAGER_H

#define CPU_KERNEL_TYPE void(void*, uint64_t, uint64_t, const void*)

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Passes/OptimizationLevel.h>

#include "saot/QuantumGate.h"

namespace saot {

struct KernelInfo {
  enum KernelType { CPU_Gate, CPU_Measure, GPU_Gate, GPU_Measure };
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

  static const CPUKernelGenConfig NativeDefaultF32;
  static const CPUKernelGenConfig NativeDefaultF64;
};

class KernelManager {
  std::unique_ptr<llvm::LLVMContext> llvmContext;
  std::unique_ptr<llvm::Module> llvmModule;

  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;

  std::vector<KernelInfo> _kernels;
public:
  KernelManager()
    : llvmContext(std::make_unique<llvm::LLVMContext>())
    , llvmModule(std::make_unique<llvm::Module>("myModule", *llvmContext))
    , llvmJIT(nullptr)
    , _kernels() {}

  const std::vector<KernelInfo>& kernels() const { return _kernels; }

  void initJIT(llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0);

  bool isJITed() const {
    assert(llvmContext == nullptr ^ llvmModule != nullptr);
    assert(llvmContext == nullptr ^ llvmJIT == nullptr);
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

  void applyCPUKernel(void* sv, int nQubits, const std::string& funcName);

  void applyCPUKernelMultithread(
    void* sv, int nQubits, const std::string& funcName, int nThreads);

};

} // namespace saot

#endif // SIMULATION_KERNELMANAGER_H
