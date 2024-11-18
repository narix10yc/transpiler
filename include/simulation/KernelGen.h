#ifndef SIMULATION_KERNELGEN_H
#define SIMULATION_KERNELGEN_H

#include <LLVM/IR/Module.h>
#include <memory>

namespace saot {

struct CPUKernelGenConfig {
    enum AmpFormat { AltFormat, SepFormat };

    int simdS                       = 2;
    int precision                   = 64;
    AmpFormat ampFormat             = AltFormat;   
    bool useFMA                     = true;
    bool useFMS                     = true;
    // parallel bits deposite from BMI2
    bool usePDEP                    = false;
    bool loadMatrixInEntry          = true;
    bool loadVectorMatrix           = false;
    bool forceDenseKernel           = false;
    double zeroSkipThres            = 1e-8;
    double shareMatrixElemThres     = 0.0;
    bool shareMatrixElemUseImmValue = false;

    static const CPUKernelGenConfig NativeDefault;
};

enum ScalarKind : int {
    SK_Zero = 0,
    SK_One = 1,
    SK_MinusOne = -1,
    SK_General = 2,
    SK_ImmValue = 3,
};

llvm::Function* genCPUCode(llvm::Module& llvmModule,
                           const CPUKernelGenConfig& config,
                           const std::vector<ScalarKind>& sigMat,
                           const std::vector<int>& qubits,
                           const std::string& funcName);


} // namespace saot

#endif // SIMULATION_KERNELGEN_H