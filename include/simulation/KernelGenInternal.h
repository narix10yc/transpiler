#ifndef SIMULATION_KERNEL_GEN_INTERNAL_H
#define SIMULATION_KERNEL_GEN_INTERNAL_H

#include <llvm/IR/IRBuilder.h>

#include <vector>

namespace saot::internal {

enum ScalarKind : int {
    SK_Zero = 0,
    SK_One = 1,
    SK_MinusOne = -1,
    SK_General = 2,
    SK_ImmValue = 3,
};

// std::vector<ScalarKind> computeSignatureMatrix();

}; // namespace saot::internal

#endif // SIMULATION_KERNEL_GEN_INTERNAL_H