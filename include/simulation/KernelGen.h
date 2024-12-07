#ifndef SIMULATION_KERNELGEN_H
#define SIMULATION_KERNELGEN_H

#include <llvm/IR/Module.h>
#include <memory>

namespace saot {

class QuantumGate;

struct CPUKernelGenConfig {
  enum AmpFormat { AltFormat, SepFormat };
  enum MatrixLoadMode { UseMatImmValues, StackLoadMatElems, StackLoadMatVecs };

  int simdS = 2;
  int precision = 64;
  AmpFormat ampFormat = AltFormat;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposite from BMI2
  bool usePDEP = false;
  bool useImmValues = true;
  bool loadMatrixInEntry = true;
  bool loadVectorMatrix = false;
  bool forceDenseKernel = false;
  double zeroSkipThres = 1e-8;
  double shareMatrixElemThres = 0.0;
  double oneTol = 1e-8;
  bool shareMatrixElemUseImmValue = false;
  MatrixLoadMode matrixLoadMode = UseMatImmValues;

  static const CPUKernelGenConfig NativeDefaultF32;
  static const CPUKernelGenConfig NativeDefaultF64;
};

/// @return A function that takes in 4 arguments: (void*, uint64_t, uint64_t,
/// void*). Pointer to statevector array, taskID begin, taskID end, and pointer
/// to matrix array (could be null).
llvm::Function* genCPUCode(llvm::Module& llvmModule,
                           const CPUKernelGenConfig& config,
                           const QuantumGate& gate,
                           const std::string& funcName);

} // namespace saot

#endif // SIMULATION_KERNELGEN_H