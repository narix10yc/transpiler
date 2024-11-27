#ifndef SIMULATION_KERNEL_GEN_INTERNAL_H
#define SIMULATION_KERNEL_GEN_INTERNAL_H

#include "saot/ScalarKind.h"
#include <llvm/IR/IRBuilder.h>

#include <vector>
#include <tuple>

namespace saot::internal {

enum FusedOpKind {
  FO_None,      // do not use fused operations
  FO_FMA_Only,  // use fma only
  FO_FMA_FMS,   // use fma and fms
};

// genMulAdd: generate multiply-add operation a * b + c.
// @param b cannot be nullptr
// @return a * b + c
llvm::Value* genMulAdd(
    llvm::IRBuilder<>& B, llvm::Value* a, llvm::Value* b, llvm::Value* c,
    ScalarKind aKind, const llvm::Twine& name = "");

// genFMA: generate negate-multiply-add operation -a * b + c.
// @param b cannot be nullptr
// @return -a * b + c
llvm::Value* genNegMulAdd(
    llvm::IRBuilder<>& B, llvm::Value* a, llvm::Value* b, llvm::Value* c,
    ScalarKind aKind, const llvm::Twine& name = "");

std::pair<llvm::Value*, llvm::Value*> genComplexInnerProduct(
    llvm::IRBuilder<>& B, const std::vector<llvm::Value*>& aVec,
    const std::vector<llvm::Value*>& bVec, const llvm::Twine& name = "",
    FusedOpKind foKind = FO_FMA_FMS);

}; // namespace saot::internal

#endif // SIMULATION_KERNEL_GEN_INTERNAL_H