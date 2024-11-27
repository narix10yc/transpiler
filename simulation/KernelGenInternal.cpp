#include "simulation/KernelGenInternal.h"

using namespace saot;
using namespace llvm;

Value* saot::internal::genMulAdd(
    IRBuilder<> &B, Value* a, Value* b, Value* c,
    ScalarKind aKind, const Twine& name) {
  assert(b && "operand b cannot be null when calling genMulAdd");
  switch (aKind) {
  case SK_General: {
    // a * b + c
    assert(a && "General kind 'a' operand cannot be null");
    if (c)
      return B.CreateIntrinsic(
        a->getType(), Intrinsic::fmuladd, {a, b, c}, nullptr, name);
    return B.CreateFMul(a, b, name);
  }
  case SK_One: {
    // b + c
    if (c)
      return B.CreateFAdd(b, c, name);
    return b;
  }
  case SK_MinusOne: {
    // -b + c
    if (c)
      return B.CreateFSub(c, b, name);
    return B.CreateFNeg(b, name);
  }
  case SK_Zero:
    return c;
  default:
    llvm_unreachable("Unknown ScalarKind");
    return nullptr;
  }
}

Value* saot::internal::genNegMulAdd(
    IRBuilder<> &B, Value* a, Value* b, Value* c,
    ScalarKind aKind, const Twine& name) {
  assert(b && "operand b cannot be null when calling genNegMulAdd");
  // special-a cases
  switch (aKind) {
  case SK_One: {
    // -b + c
    if (c)
      return B.CreateFSub(c, b, name);
    return B.CreateFNeg(b, name);
  }
  case SK_MinusOne: {
    // b + c
    if (c)
      return B.CreateFAdd(b, c, name);
    return b;
  }
  case SK_Zero:
    return c;
  default:
    break;
  }
  assert(aKind == SK_General);
  assert(a && "General kind 'a' operand cannot be null");

  // -a * b + c
  auto* aNeg = B.CreateFNeg(a, "a.neg");
  if (c)
    return B.CreateIntrinsic(
      aNeg->getType(), Intrinsic::fmuladd, {aNeg, b, c}, nullptr, name);
  return B.CreateMul(aNeg, b, name);
}


std::pair<Value*, Value*> saot::internal::genComplexInnerProduct(
    IRBuilder<>& B,
    const std::vector<Value*>& aVec, const std::vector<Value*>& bVec,
    const Twine& name, saot::internal::FusedOpKind foKind) {
  assert(aVec.size() == bVec.size());
  unsigned size = aVec.size();
  Value* re;
  Value* im;

  assert(0 && "Not implemented");
  return {nullptr, nullptr};
}
