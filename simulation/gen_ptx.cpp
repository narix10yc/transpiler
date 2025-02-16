#include "simulation/ir_generator.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

#include <algorithm>
#include <bitset>
#include <cmath>

using namespace IOColor;
using namespace llvm;
using namespace simulation;
using namespace cast;

namespace {

Value* genOptFMul(Value* a, Value* b, ScalarKind aKind, IRBuilder<>& builder) {
  switch (aKind) {
  case SK_General:
    assert(a);
    return builder.CreateFMul(a, b);
  case SK_One:
    return b;
  case SK_MinusOne:
    return builder.CreateFNeg(b);
  case SK_Zero:
    return nullptr;
  default:
    llvm_unreachable("Unknown ScalarKind");
    return nullptr;
  }
}

// return a * b + c
Value* genMulAndAdd(Value* a, Value* b, Value* c, ScalarKind aKind,
                    IRBuilder<>& builder) {
  assert(b);

  switch (aKind) {
  case SK_General:
    assert(a);
    if (c)
      return builder.CreateIntrinsic(a->getType(), Intrinsic::fmuladd,
                                     {a, b, c});
    return builder.CreateFMul(a, b);
  case SK_One:
    if (c)
      return builder.CreateFAdd(b, c);
  case SK_MinusOne:
    if (c)
      return builder.CreateFSub(c, b);
    return builder.CreateFNeg(b);
  case SK_Zero:
    return c;
  default:
    llvm_unreachable("Unknown ScalarKind");
    return nullptr;
  }
}

} // namespace

Function* IRGenerator::generateCUDAKernel(const QuantumGate& gate,
                                          const CUDAGenerationConfig& config,
                                          const std::string& funcName) {
  const auto& qubits = gate.qubits;
  const int nQubits = qubits.size();

  auto* gateCMat = gate.gateMatrix.getConstantMatrix();
  // printConstantMatrix(std::cerr, *gateCMat);

  Type* scalarTy =
      (config.precision == 32) ? builder.getFloatTy() : builder.getDoubleTy();
  Function* func;
  Argument* pSvArg, *pMatArg;
  { /* function declaration */

    /*
        Address space:
        0: Generic;
        1: Global;
        2: Internal Use;
        3: Shared;
        4: Constant (often 64KB)
        5: Local;

        For a reference see https://llvm.org/docs/NVPTXUsage.html#id32
    */
    SmallVector<Type*> argType{
        builder.getPtrTy(1U),
        builder.getPtrTy(config.useConstantMemSpaceForMatPtrArg ? 4U : 1U),
    };

    auto* funcType = FunctionType::get(builder.getVoidTy(), argType, false);
    func = Function::Create(funcType, Function::ExternalLinkage, funcName,
                            getModule());
    if (funcName == "") {
      std::stringstream ss;
      ss << "ptx_kernel_";
      func->setName(ss.str());
    } else
      func->setName(funcName);

    pSvArg = func->getArg(0);
    pSvArg->setName("p.sv");
    pMatArg = func->getArg(1);
    pMatArg->setName("p.mat");

    // mark this function as a kernel
    auto* mdString = MDString::get(*getContext(), "kernel");
    auto* mdOne = ConstantAsMetadata::get(builder.getInt32(1));
    auto* kernelMetadata = MDNode::get(
       * getContext(), {ValueAsMetadata::get(func), mdString, mdOne});
    getModule()
        ->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(kernelMetadata);
  } // end function declearation

  Value* counterV;

  BasicBlock* entryBB = BasicBlock::Create(*_context, "entry", func);
  builder.SetInsertPoint(entryBB);
  auto* threadIdx = builder.CreateIntrinsic(builder.getInt32Ty(),
                                            Intrinsic::nvvm_read_ptx_sreg_tid_x,
                                            {}, nullptr, "threadIdx");
  auto* nthreads = builder.CreateIntrinsic(builder.getInt32Ty(),
                                           Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                           {}, nullptr, "nthreads");
  auto* blockIdx = builder.CreateIntrinsic(
      builder.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, nullptr,
      "blockIdx");
  counterV = builder.CreateMul(blockIdx, nthreads);
  counterV = builder.CreateAdd(counterV, threadIdx, "counter.i32");
  counterV = builder.CreateIntCast(counterV, builder.getInt64Ty(), true,
                                   "global.thread.idx");
  /*
  Example: with target qubits 2, 4, 5
  counter:   xxxhgfedcba
  pbex mask: 11111001011
  idxStart:  hgfed00c0ba

  hgfed00c0ba = (xxxhgfedcba & 00000000011) << 0
              + (xxxhgfedcba & 00000000100) << 1
              + (xxxhgfedcba & 11111111000) << 3

  We build this segment by segment. For [2, 4, 5], there are 3 segments:
      [0, 2),      [3, 4),      [5, ),
  corresponding to mask
      00000000011, 00000000100, 11111111000
  */

  // utils::printVector(qubits, std::cerr << "target qubits: ") << "\n";

  // the pointer of sv start in this thread
  Value* svPtrV;
  {
    Value* idxStartV = builder.getInt64(0ULL);
    Value* tmpCounterV;
    uint64_t mask = 0ULL;
    int highestQ = qubits.back();
    int qIdx = 0;
    int counterQ = 0;
    for (int q = 0; q <= highestQ; q++) {
      if (q < qubits[qIdx]) {
        mask |= (1 << counterQ++);
        continue;
      }
      // q == qubits[qIdx];
      ++qIdx;
      if (mask == 0)
        continue;
      tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
      tmpCounterV = builder.CreateShl(tmpCounterV, (qIdx - 1), "tmpCounter");
      idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
      // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ") <<
      // " << (qIdx - 1) << "\n";
      mask = 0ULL;
    }
    mask = ~((1ULL << (qubits.back() - nQubits + 1)) - 1);
    // std::cerr << "  (globalThreadIdx & " << utils::as0b(mask, 32) << ") << "
    // << (nQubits) << "\n";

    tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = builder.CreateShl(tmpCounterV, nQubits, "tmpCounter");
    idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");
    idxStartV = builder.CreateShl(idxStartV, 1, "idxStart");
    svPtrV = builder.CreateGEP(scalarTy, pSvArg, idxStartV, "sv.ptr");
  }

  uint64_t N = 1 << nQubits;
  // std::vector<Value*> reMats(N), imMats(N), reAmpPtrs(N), imAmpPtrs(N),
  // reAmps(N), imAmps(N);
  std::vector<Value*> reAmpPtrs(N), imAmpPtrs(N), reAmps(N), imAmps(N);
  auto sigMat =
      gate.gateMatrix.getSignatureMatrix(config.zeroTol, config.oneTol);
  if (config.forceDenseKernel) {
    for (auto& sig : sigMat) {
      sig.real(SK_General);
      sig.imag(SK_General);
    }
  }

  // load amplitude set
  for (uint64_t i = 0; i < N; i++) {
    uint64_t delta = 0ULL;
    for (int b = 0; b < nQubits; b++) {
      if (i & (1 << b))
        delta |= (1 << qubits[b]);
    }
    // std::cerr << "amp idx " << utils::as0b(i, nQubits) << ", delta = " <<
    // utils::as0b(delta, 32) << "\n";

    reAmpPtrs[i] = builder.CreateConstGEP1_64(
        scalarTy, svPtrV, 2 * delta, "amp.re.ptr." + std::to_string(i));
    reAmps[i] = builder.CreateLoad(scalarTy, reAmpPtrs[i],
                                   "amp.re." + std::to_string(i));
    imAmpPtrs[i] = builder.CreateConstGEP1_64(
        scalarTy, svPtrV, 2 * delta + 1, "amp.im.ptr." + std::to_string(i));
    imAmps[i] = builder.CreateLoad(scalarTy, imAmpPtrs[i],
                                   "amp.im." + std::to_string(i));
  }

  for (int r = 0; r < N; r++) {
    // for (int c = 0; c < N; c++) {
    //     // load the r-th row of the matrix
    //     std::string suffix = std::to_string(r) + "." + std::to_string(c);
    //     auto* idxReV = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL* 
    //     (N*r + c), "idx.mat.re." + suffix); reMats[c] =
    //     builder.CreateLoad(scalarTy, idxReV, "mat.re." + suffix); auto*
    //     idxImV = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL * (N*r +
    //     c) + 1, "idx.mat.im." + suffix); imMats[c] =
    //     builder.CreateLoad(scalarTy, idxImV, "mat.im." + suffix);
    // }

    // matrix-vector multiplication
    // Value* updatedReAmp = reAmps[r];
    // Value* updatedImAmp = imAmps[r];
    // for (int c = 0; c < N; c++) {
    //     auto pair = genComplexMultiply({updatedReAmp, updatedImAmp},
    //     {reMats[c], imMats[c]}); updatedReAmp = pair.first; updatedImAmp =
    //     pair.second;
    // }
    // store back
    // builder.CreateStore(updatedReAmp, reAmpPtrs[r]);
    // builder.CreateStore(updatedImAmp, imAmpPtrs[r]);

    // mat-vec mul alternative version
    // auto updated = genComplexDotProduct(reAmps, imAmps, reMats, imMats);
    // builder.CreateStore(updated.first, reAmpPtrs[r]);
    // builder.CreateStore(updated.second, imAmpPtrs[r]);

    // Alternative version, potentially having better locality
    // updatedReAmp = sum(reAmps_i * reMats_i) - sum(imAmps_i * imMats_i)
    // updatedImAmp = sum(reAmps_i * imMats_i) + sum(imAmps_i * reMats_i)
    Value* reMatPtr, *imMatPtr, *reMat, *imMat;
    if (sigMat(0, 0).real() == SK_General) {
      if (config.useImmValues && gateCMat) {
        reMat = ConstantFP::get(scalarTy, gateCMat->rc(0, 0).real());
      } else {
        reMatPtr = builder.CreateConstGEP1_64(
            scalarTy, pMatArg, 0ULL, "mat.re.ptr." + std::to_string(r) + ".0");
        reMat = builder.CreateLoad(scalarTy, reMatPtr,
                                   "mat.re." + std::to_string(r) + ".0");
      }
    }
    if (sigMat(0, 0).imag() == SK_General) {
      if (config.useImmValues && gateCMat) {
        imMat = ConstantFP::get(scalarTy, gateCMat->rc(0, 0).imag());
      } else {
        imMatPtr = builder.CreateConstGEP1_64(
            scalarTy, pMatArg, 1ULL, "mat.im.ptr." + std::to_string(r) + ".0");
        imMat = builder.CreateLoad(scalarTy, imMatPtr,
                                   "mat.im." + std::to_string(r) + ".0");
      }
    }

    Value* updatedReAmp0 =
        genOptFMul(reMat, reAmps[0], sigMat(0, 0).real(), builder);
    Value* updatedReAmp1 =
        genOptFMul(imMat, imAmps[0], sigMat(0, 0).imag(), builder);
    Value* updatedImAmp =
        genOptFMul(reMat, imAmps[0], sigMat(0, 0).real(), builder);
    updatedImAmp = genMulAndAdd(imMat, reAmps[0], updatedImAmp,
                                sigMat(0, 0).imag(), builder);
    for (int c = 1; c < N; c++) {
      std::string suffix = std::to_string(r) + "." + std::to_string(c);
      reMat = nullptr;
      imMat = nullptr;

      size_t matIdx = r * N + c;
      if (sigMat(r, c).real() == SK_General) {
        if (config.useImmValues && gateCMat) {
          reMat = ConstantFP::get(scalarTy, gateCMat->rc(r, c).real());
        } else {
          reMatPtr = builder.CreateConstGEP1_64(
              scalarTy, pMatArg, 2ULL * matIdx, "idx.mat.re." + suffix);
          reMat = builder.CreateLoad(scalarTy, reMatPtr, "mat.re." + suffix);
        }
      }
      if (sigMat(r, c).imag() == SK_General) {
        if (config.useImmValues && gateCMat) {
          imMat = ConstantFP::get(scalarTy, gateCMat->rc(r, c).imag());
        } else {
          imMatPtr = builder.CreateConstGEP1_64(
              scalarTy, pMatArg, 2ULL * matIdx + 1, "idx.mat.im." + suffix);
          imMat = builder.CreateLoad(scalarTy, imMatPtr, "mat.im." + suffix);
        }
      }

      // updatedReAmp0 = builder.CreateIntrinsic(
      //     scalarTy, Intrinsic::fmuladd, { reAmps[c], reMat, updatedReAmp0 });
      // updatedReAmp1 = builder.CreateIntrinsic(
      //     scalarTy, Intrinsic::fmuladd, { imAmps[c], imMat, updatedReAmp1 });
      // updatedImAmp = builder.CreateIntrinsic(
      //     scalarTy, Intrinsic::fmuladd, { reAmps[c], imMat, updatedImAmp });
      // updatedImAmp = builder.CreateIntrinsic(
      //     scalarTy, Intrinsic::fmuladd, { imAmps[c], reMat, updatedImAmp });

      updatedReAmp0 = genMulAndAdd(reMat, reAmps[c], updatedReAmp0,
                                   sigMat(r, c).real(), builder);
      updatedReAmp1 = genMulAndAdd(imMat, imAmps[c], updatedReAmp1,
                                   sigMat(r, c).imag(), builder);
      updatedImAmp = genMulAndAdd(reMat, imAmps[c], updatedImAmp,
                                  sigMat(r, c).real(), builder);
      updatedImAmp = genMulAndAdd(imMat, reAmps[c], updatedImAmp,
                                  sigMat(r, c).imag(), builder);
    }

    Value* updatedReAmp = nullptr;
    if (updatedReAmp0 && updatedReAmp1)
      updatedReAmp = builder.CreateFSub(updatedReAmp0, updatedReAmp1);
    else if (updatedReAmp0)
      updatedReAmp = updatedReAmp0;
    else if (updatedReAmp1)
      updatedReAmp = builder.CreateFNeg(updatedReAmp1);
    else
      llvm_unreachable("updatedReAmp should not be zero");

    builder.CreateStore(updatedReAmp, reAmpPtrs[r]);
    builder.CreateStore(updatedImAmp, imAmpPtrs[r]);
  }

  builder.CreateRetVoid();
  return func;
}
