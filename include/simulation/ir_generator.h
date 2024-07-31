#ifndef SIMULATION_CODEGEN_H_
#define SIMULATION_CODEGEN_H_

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include <vector>
#include <array>

#include "quench/QuantumGate.h"

namespace simulation {

struct IRGeneratorConfig {
    int simdS;
    int realTy;
    bool useFMA;
    bool useFMS;
    bool usePDEP; // parallel bits deposite from BMI2
    bool loadMatrixInEntry;
    bool loadVectorMatrix;
    int verbose;
    
    // bool prefetchEnable;
    // int prefetchDistance;
    static IRGeneratorConfig DefaultX86F64() {
        return {
            .simdS = 2,
            .realTy = 64,
            .useFMA = true,
            .useFMS = true,
            .usePDEP = true,
            .loadMatrixInEntry = true,
            .loadVectorMatrix = false,
            .verbose = 0,
        };
    }

    static IRGeneratorConfig DefaultX86F32() {
        return {
            .simdS = 3,
            .realTy = 32,
            .useFMA = true,
            .useFMS = true,
            .usePDEP = true,
            .loadMatrixInEntry = true,
            .loadVectorMatrix = false,
            .verbose = 0,
        };
    }

    static IRGeneratorConfig DefaultARMF64() {
        return {
            .simdS = 1,
            .realTy = 64,
            .useFMA = true,
            .useFMS = true,
            .usePDEP = false,
            .loadMatrixInEntry = true,
            .loadVectorMatrix = false,
            .verbose = 0,
        };
    }

    static IRGeneratorConfig DefaultARMF32() {
        return {
            .simdS = 2,
            .realTy = 32,
            .useFMA = true,
            .useFMS = true,
            .usePDEP = false,
            .loadMatrixInEntry = true,
            .loadVectorMatrix = false,
            .verbose = 0,
        };
    }
};


struct PrefetchConfiguration {
    bool enable;
    int distance;
};

/// @brief IR Generator.
/// @param vecSizeInBits: Required; default 2; the value of s.
/// @param useFMA: default true; whether use fused multiplication-addition.
/// @param realTy: default f64; type of real scalar (f32 / f64)
/// @param ampFormat: default separate; format of statevector amplitude
///                   (separate / alternating)
class IRGenerator {
    llvm::LLVMContext llvmContext;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::Module> mod;

public:
    enum class RealTy {
        Float, Double
    };

public:
    unsigned vecSizeInBits;
    int chunkC; // Not Implemented
    bool useFMA;
    bool useFMS;
    bool usePDEP; // parallel bits deposite from BMI2
    bool loadMatrixInEntry;
    bool loadVectorMatrix;
    bool allocaMatrixInStack;
    int verbose;
    bool forceDenseKernel;
    RealTy realTy;
    PrefetchConfiguration prefetchConfig;

    // IRGeneratorConfig config;

public:
    IRGenerator(unsigned vecSizeInBits=2, const std::string& moduleName = "myModule") : 
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>(moduleName, llvmContext)),
        vecSizeInBits(vecSizeInBits),
        chunkC(-1),
        useFMA(true),
        useFMS(true),
        usePDEP(true),
        loadMatrixInEntry(true),
        loadVectorMatrix(true),
        allocaMatrixInStack(true),
        verbose(0),
        forceDenseKernel(false),
        realTy(RealTy::Double),
        prefetchConfig({.enable=false, .distance = 1}) {}

    const llvm::Module& getModule() const { return *mod; }
    llvm::Module& getModule() { return *mod; }

    void loadFromFile(const std::string& fileName);
    
    /// @brief Generate the IR that applies new_aa = aa + bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag special values are +1, -1, or 0
    /// @return aa + bb * cc. Possible nullptr, when aa is nullptr and bbFlag = 0
    llvm::Value* genMulAdd(
            llvm::Value* aa,
            llvm::Value* bb,
            llvm::Value* cc,
            int bbFlag,
            const llvm::Twine& bbccName="",
            const llvm::Twine& aaName="");

    /// @brief Generate the IR that applies new_aa = aa - bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to -bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag special values are +1, -1, or 0
    /// @return aa - bb * cc
    llvm::Value* genMulSub(
            llvm::Value* aa,
            llvm::Value* bb,
            llvm::Value* cc,
            int bbFlag,
            const llvm::Twine& bbccName="",
            const llvm::Twine& aaName="");

    llvm::Function*
    generateKernel(const quench::quantum_gate::QuantumGate& gate,
                   const std::string& funcName = "");

    llvm::Function*
    generateAlternatingKernel(const quench::quantum_gate::QuantumGate& gate,
                              const std::string& funcName = "");
};


} // namespace simulation

#endif // SIMULATION_CODEGEN_H_