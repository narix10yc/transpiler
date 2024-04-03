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

class IRGenerator {
    llvm::LLVMContext llvmContext;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::Module> mod;

    // parameters
    unsigned vecSizeInBits;
private:
    llvm::Value* getVectorWithSameElem(llvm::Type* realTy, const unsigned length, 
            llvm::Value* elem, const llvm::Twine &name = "") {
        llvm::Value* vec = llvm::UndefValue::get(llvm::VectorType::get(realTy, length, false));
        for (size_t i = 0; i < length - 1; ++i) {
            vec = builder.CreateInsertElement(vec, elem, i, name + "_insert_" + std::to_string(i));
        }
        vec = builder.CreateInsertElement(vec, elem, length - 1, name + "_vec");  
        return vec;  
    }

    llvm::Value* getVectorWithSameElem(llvm::Type* realTy, const unsigned length, 
            const double value, const llvm::Twine &name = "") {
        auto* elem = llvm::ConstantFP::get(llvmContext, llvm::APFloat(value));
        return getVectorWithSameElem(realTy, length, elem, name);
    }

public:
    IRGenerator() : 
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>("myModule", llvmContext)) {
        vecSizeInBits = 2;
    }

    void setVectorSizeInBits(unsigned bits) {
        vecSizeInBits = bits;
    }

    enum RealTy : int { Float, Double };
    
    const llvm::Module& getModule() const { return *mod; }

    /// @brief Generate the IR that applies new_aa = aa +/- bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag either Add (for bb = 1), Sub (for bb = -1), Zero (for bb 
    ///   = 0), or Normal (otherwise)
    /// @return aa + bb * cc
    llvm::Value* genMulAddOrMulSub(
            llvm::Value* aa,
            bool add,
            llvm::Value* bb,
            llvm::Value* cc,
            int bbFlag,
            const llvm::Twine& bbccName = "",
            const llvm::Twine& aaName = "");

    void genU3(const int64_t k, 
            const llvm::StringRef funcName, 
            const RealTy realType,
            std::optional<double> theta, 
            std::optional<double> phi, 
            std::optional<double> lambd, 
            double thres=1e-8);
};


} // namespace simulation

#endif // SIMULATION_CODEGEN_H_