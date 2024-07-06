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

class GeneratorConfiguration {
public:
    unsigned s;
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
    bool useFMA;
    int verbose;
    RealTy realTy;

public:
    IRGenerator(unsigned vecSizeInBits=2) : 
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>("myModule", llvmContext)),
        vecSizeInBits(vecSizeInBits),
        useFMA(true),
        realTy(RealTy::Double),
        verbose(0) {}

    const llvm::Module& getModule() const { return *mod; }

    void setUseFMA(bool b) { useFMA = b; }
    void setRealTy(RealTy ty) { realTy = ty; }
    void setVerbose(int v) { verbose = v; }

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
                   const std::string& funName = "");
};


} // namespace simulation

#endif // SIMULATION_CODEGEN_H_