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

#include <memory>
#include <vector>
#include <array>

#include "simulation/types.h"

namespace simulation {

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
    unsigned vecSizeInBits;
    bool useFMA;
    ir::RealTy realTy;
    ir::AmpFormat ampFormat;
private:
    llvm::Value* 
    genVectorWithSameElem(llvm::Type* elemTy, unsigned length, 
                          llvm::Value* elem, const llvm::Twine &name = "") {
        llvm::Type* vecTy = llvm::VectorType::get(elemTy, length, false);
        llvm::Value* vec = llvm::UndefValue::get(vecTy);
        for (size_t i = 0; i < length - 1; ++i) {
            vec = builder.CreateInsertElement(vec, elem, i, name + "_insert_" + std::to_string(i));
        }
        vec = builder.CreateInsertElement(vec, elem, length - 1, name + "_vec");  
        return vec;  
    }

    llvm::Function* genU3_Sep(const ir::U3Gate& u3, std::string funcName="");
    llvm::Function* genU3_Alt(const ir::U3Gate& u3, std::string funcName="");

public:
    IRGenerator(unsigned vecSizeInBits=2) : 
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>("myModule", llvmContext)),
        vecSizeInBits(vecSizeInBits),
        useFMA(true),
        realTy(ir::RealTy::Double),
        ampFormat(ir::AmpFormat::Separate) {}

    const llvm::Module& getModule() const { return *mod; }

    void setUseFMA(bool b) { useFMA = b; }
    void setRealTy(ir::RealTy ty) { realTy = ty; }
    void setAmpFormat(ir::AmpFormat format) { ampFormat = format; }
    
    /// @brief Generate the IR that applies new_aa = aa + bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag special values are +1, -1, or 0
    /// @return aa + bb * cc
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

    llvm::Function* genU3(const ir::U3Gate& u3, std::string funcName="") {
        if (ampFormat == ir::AmpFormat::Separate)
            return genU3_Sep(u3, funcName);
        else
            return genU3_Alt(u3, funcName);
    }
    
    llvm::Function* genU2q(const U2qGate& u2q,
                           std::string funcName="");
};


} // namespace simulation

#endif // SIMULATION_CODEGEN_H_