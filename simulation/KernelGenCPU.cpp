#define DEBUG_TYPE "cpu-codegen"
#include "llvm/Support/Debug.h"

#include "saot/QuantumGate.h"
#include "simulation/KernelGen.h"
#include "simulation/KernelGenInternal.h"

#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/IR/IntrinsicsX86.h"

#include <algorithm>
#include <bitset>
#include <cmath>

using namespace IOColor;
// using namespace utils;
using namespace llvm;
using namespace saot;

struct matrix_data_t {
  Value* realVal;
  Value* imagVal;
  int realFlag;
  int imagFlag;
  bool realLoadNeg;
  bool imagLoadNeg;
};

namespace {
/// v0 and v1 are always sorted. Perform linear time merge
/// @return (mask, vec)
std::pair<std::vector<int>, std::vector<int>>
getMaskToMerge(const std::vector<int> &v0, const std::vector<int> &v1) {
  assert(v0.size() == v1.size());
  const auto s = v0.size();
  std::vector<int> mask(2 * s);
  std::vector<int> vec(2 * s);
  unsigned i0 = 0, i1 = 0, i;
  int elem0, elem1;
  while (i0 < s || i1 < s) {
    i = i0 + i1;
    if (i0 == s) {
      vec[i] = v1[i1];
      mask[i] = i1 + s;
      i1++;
      continue;
    }
    if (i1 == s) {
      vec[i] = v0[i0];
      mask[i] = i0;
      i0++;
      continue;
    }
    elem0 = v0[i0];
    elem1 = v1[i1];
    if (elem0 < elem1) {
      vec[i] = elem0;
      mask[i] = i0;
      i0++;
    } else {
      vec[i] = elem1;
      mask[i] = i1 + s;
      i1++;
    }
  }
  return std::make_pair(mask, vec);
}

struct CPUArgs {
  Argument* pSvArg;       // ptr to statevector
  Argument* ctrBeginArg;  // counter begin
  Argument* ctrEndArg;    // counter end
  Argument* pMatArg;      // ptr to matrix
};

inline Function* cpuGetFunctionDeclaration(
    IRBuilder<>& B, Module& M, const std::string& funcName,
    const CPUKernelGenConfig& config, CPUArgs& args) {
 auto argType = SmallVector<Type*>{
  B.getPtrTy(), B.getInt64Ty(), B.getInt64Ty(), B.getPtrTy()};

  auto* funcTy = FunctionType::get(B.getVoidTy(), argType, false);
  auto* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, M);

  args.pSvArg = func->getArg(0);
  args.pSvArg->setName("p.sv.arg");
  args.ctrBeginArg = func->getArg(1);
  args.ctrBeginArg->setName("ctr.begin");
  args.ctrEndArg = func->getArg(2);
  args.ctrEndArg->setName("ctr.end");
  args.pMatArg = func->getArg(3);
  args.pMatArg->setName("pmat");

  return func;
}

struct MatData {
  Value* reVal;
  Value* imVal;
  ScalarKind reKind;
  ScalarKind imKind;
};

inline std::vector<MatData> cpuGetMatData(
    IRBuilder<>& B, const GateMatrix& gateMatrix,
    const CPUKernelGenConfig& config) {
  int k = gateMatrix.nqubits();
  unsigned K = 1 << k;
  unsigned KK = K * K;

  std::vector<MatData> data;
  data.resize(KK);

  double zTol = config.zeroSkipThres / K;
  double oTol = config.oneTol / K;
  double sTol = config.shareMatrixElemThres / K;
  const auto* cMat = gateMatrix.getConstantMatrix();
  assert(cMat && "Parametrized matrices not implemented yet");

  for (unsigned i = 0; i < KK; i++) {
    if (cMat == nullptr || config.forceDenseKernel) {
      data[i].reKind = SK_General;
      data[i].imKind = SK_General;
      continue;
    }
    auto real = cMat->data[i].real();
    auto imag = cMat->data[i].imag();

    if (std::abs(real) < zTol)
      data[i].reKind = SK_Zero;
    else if (std::abs(real - 1.0) < oTol)
      data[i].reKind = SK_One;
    else if (std::abs(real + 1.0) < oTol)
      data[i].reKind = SK_MinusOne;
    else if (config.useImmValues) {
      data[i].reKind = SK_ImmValue;
      data[i].reVal = ConstantFP::get(B.getContext(), APFloat(real));
    }
    else
      data[i].reKind = SK_General;

    if (std::abs(imag) < zTol)
      data[i].imKind = SK_Zero;
    else if (std::abs(imag - 1.0) < oTol)
      data[i].imKind = SK_One;
    else if (std::abs(imag + 1.0) < oTol)
      data[i].imKind = SK_MinusOne;
    else if (config.useImmValues) {
      data[i].imKind = SK_ImmValue;
      data[i].imVal = ConstantFP::get(B.getContext(), APFloat(imag));
    }
    else
      data[i].imKind = SK_General;
  }
  return data;
}

} // anonymous namespace

Function* saot::genCPUCode(llvm::Module &llvmModule,
                           const CPUKernelGenConfig &config,
                           const QuantumGate &gate,
                           const std::string &funcName) {
  const unsigned s = config.simdS;
  const unsigned S = 1ULL << s;
  const unsigned k = gate.qubits.size();
  const unsigned K = 1ULL << k;
  const unsigned KK = K * K;

  auto &llvmContext = llvmModule.getContext();

  IRBuilder<> B(llvmContext);
  Type* scalarTy = (config.precision == 32) ? B.getFloatTy() : B.getDoubleTy();
  
  // init function
  CPUArgs args;
  Function* func = cpuGetFunctionDeclaration(B, llvmModule, funcName, config, args);

  // init matrix
  auto matrixData = cpuGetMatData(B, gate.gateMatrix, config);

  // init basic blocks
  BasicBlock* entryBB = BasicBlock::Create(llvmContext, "entry", func);
  BasicBlock* loopBB = BasicBlock::Create(llvmContext, "loop", func);
  BasicBlock* loopBodyBB = BasicBlock::Create(llvmContext, "loop.body", func);
  BasicBlock* retBB = BasicBlock::Create(llvmContext, "ret", func);

  // split qubits
  int sepBit;
  SmallVector<int, 8U> simdBits, hiBits, loBits;
  { /* split qubits */
  unsigned q = 0;
  auto qubitsIt = gate.qubits.cbegin();
  const auto qubitsEnd = gate.qubits.cend();
  while (simdBits.size() != s) {
    if (qubitsIt != qubitsEnd && *qubitsIt == q) {
      loBits.push_back(q);
      ++qubitsIt;
    } else {
      simdBits.push_back(q);
    }
    ++q;
  }
  while (qubitsIt != qubitsEnd) {
      hiBits.push_back(*qubitsIt);
      ++qubitsIt;
  }
  sepBit = (s == 0) ? 0 : simdBits.back() + 1;
  }

  const unsigned vecSize = 1U << sepBit;
  const unsigned vecSizex2 = vecSize << 1;
  auto* vecType = VectorType::get(scalarTy, vecSize, false);
  auto* vecTypeX2 = VectorType::get(scalarTy, vecSizex2, false);

  const unsigned lk = loBits.size();
  const unsigned LK = 1 << lk;
  const unsigned hk = hiBits.size();
  const unsigned HK = 1 << hk;

  // debug print qubit splits
  std::cerr << CYAN_FG << "-- qubit split done\n" << RESET;\
  utils::printVector(loBits, std::cerr << "- lower bits: ") << "\n";
  utils::printVector(simdBits, std::cerr << "- simd bits: ") << "\n";
  utils::printVector(hiBits, std::cerr << "- higher bits: ") << "\n";
  std::cerr << "sepBit:  " << sepBit << "\n";
  std::cerr << "vecSize: " << vecSize << "\n";
  
  B.SetInsertPoint(entryBB);
  // load matrix (if needed)
  if (config.matrixLoadMode == CPUKernelGenConfig::VectorInStack) {
    auto* matV = B.CreateLoad(
        VectorType::get(scalarTy, 2 * KK, false), args.pMatArg, "matrix");
    for (unsigned i = 0; i < KK; i++) {
      if (matrixData[i].reKind == SK_General) {
        assert(matrixData[i].reVal == nullptr);
        matrixData[i].reVal = B.CreateShuffleVector(
          matV, std::vector<int>(S, 2 * i), "m.re." + std::to_string(i));
      }
      if (matrixData[i].imKind == SK_General) {
        assert(matrixData[i].imVal == nullptr);
        matrixData[i].imVal = B.CreateShuffleVector(
          matV, std::vector<int>(S, 2 * i + 1), "m.im." + std::to_string(i));
      }
    }
  } else if (config.matrixLoadMode == CPUKernelGenConfig::ElemInStack) {
    for (unsigned i = 0; i < KK; i++) {
      if (matrixData[i].imKind == SK_General) {
        assert(matrixData[i].imVal == nullptr);
        auto* pReVal = B.CreateConstGEP1_32(
          scalarTy, args.pMatArg, 2 * i, "ptr.m.re." + std::to_string(i));
        auto* mReVal = B.CreateLoad(
          scalarTy, pReVal, "m.re." + std::to_string(i) + ".elem");                             
        matrixData[i].reVal = B.CreateVectorSplat(
          S, mReVal, "m.re." + std::to_string(i));
      }

      if (matrixData[i].imKind == SK_General) {
        assert(matrixData[i].imVal == nullptr);
        auto* pImVal = B.CreateConstGEP1_32(
            scalarTy, args.pMatArg, 2 * i + 1, "ptr.m.im." + std::to_string(i));
        auto* mImVal = B.CreateLoad(
          scalarTy, pImVal, "m.im." + std::to_string(i) + ".elem");
        matrixData[i].imVal = B.CreateVectorSplat(
          S, mImVal, "m.im." + std::to_string(i));
      }
    }
  }

  B.CreateBr(loopBB);

  // loop entry: set up counter
  B.SetInsertPoint(loopBB);
  PHINode* taskIdV = B.CreatePHI(B.getInt64Ty(), 2, "taskid");
  taskIdV->addIncoming(args.ctrBeginArg, entryBB);
  Value* cond = B.CreateICmpSLT(taskIdV, args.ctrEndArg, "cond");
  B.CreateCondBr(cond, loopBodyBB, retBB);

  // loop body
  B.SetInsertPoint(loopBodyBB);
  
  // the start pointer in the SV based on taskID
  Value* ptrSvBeginV = nullptr;
  if (hiBits.empty()) {
    ptrSvBeginV = B.CreateGEP(vecTypeX2, args.pSvArg, taskIdV, "ptr.sv.begin");
  } else {
    // the shift from args.pSvArg in the unit of vecTypeX2
    Value* idxStartV = B.getInt64(0ULL);
    Value* tmpCounterV;
    uint64_t mask = 0ULL;
    int highestQ = hiBits.back();
    int qIdx = 0;
    int counterQ = 0;
    for (int q = sepBit; q <= highestQ; q++) {
      if (q < hiBits[qIdx]) {
        mask |= (1 << counterQ++);
        continue;
      }
      ++qIdx;
      if (mask == 0)
        continue;
      tmpCounterV = B.CreateAnd(taskIdV, mask, "tmp.taskid");
      tmpCounterV = B.CreateShl(tmpCounterV, (qIdx - 1), "tmp.taskid");
      idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "tmp.idx.begin");
      std::cerr << "  (taskID & " << utils::as0b(mask, highestQ) << ") << "
                << (qIdx - 1) << "\n";
      mask = 0ULL;
    }
    mask = ~((1ULL << (highestQ - sepBit - hk + 1)) - 1);
    std::cerr << "  (taskID & " << utils::as0b(mask, 16) << ") << "
              << hk << "\n";

    tmpCounterV = B.CreateAnd(taskIdV, mask, "tmp.taskid");
    tmpCounterV = B.CreateShl(tmpCounterV, hk, "tmp.taskid");
    idxStartV = B.CreateAdd(idxStartV, tmpCounterV, "idx.begin");
    ptrSvBeginV = B.CreateGEP(vecTypeX2, args.pSvArg, idxStartV, "ptr.sv.begin");
  }


  /* load amplitude registers
    There are a total of 2K size-S amplitude registers (K real and K imag).
    In Alt Format, we load HK size-(2*S*LK) LLVM registers. i.e. Loop over
    higher qubits
    There are two stages of shuffling (splits)
    - Stage 1:
      Each size-(2*S*LK) reg is shuffled into 2 size-(S*LK) regs, the reFull
      and imFull. There are a total of (2 * HK) reFull and imFull each.
    - Stage 2:
      Each reFull (resp. imFull) is shuffled into LK size-S regs, the reAmps
      (resp. imAmps) vector.
  */

  SmallVector<int> reSplitMasks;
  SmallVector<int> imSplitMasks;
  {
  const auto size = LK * S;
  unsigned pdepMask = 0U;
  for (const auto& b : simdBits)
    pdepMask |= (1 << b);
  
  reSplitMasks.resize_for_overwrite(size);
  imSplitMasks.resize_for_overwrite(size);
  for (unsigned i = 0; i < size; i++) {
    reSplitMasks[i] = utils::pdep32(i, pdepMask);
    imSplitMasks[i] = reSplitMasks[i] | (1 << s);
  }
  std::cerr << "- reSplitMasks: [";
  for (const auto& e : reSplitMasks)
    std::cerr << utils::as0b(e, sepBit + 1) << ",";
  std::cerr << "]\n";
  std::cerr << "- imSplitMasks: [";
  for (const auto& e : imSplitMasks)
    std::cerr << utils::as0b(e, sepBit + 1) << ",";
  std::cerr << "]\n";
  }

  // load vectors
  SmallVector<Value*> reAmps; // real amplitudes
  SmallVector<Value*> imAmps; // imag amplitudes
  reAmps.resize_for_overwrite(K);
  imAmps.resize_for_overwrite(K);

  SmallVector<Value*> pSvs;
  pSvs.resize_for_overwrite(HK);
  for (unsigned hi = 0; hi < HK; hi++) {
    uint64_t idxShift = 0ULL;
    for (unsigned hbit = 0; hbit < hk; hbit++) {
      if (hi & (1 << hbit))
        idxShift += 1ULL << hiBits[hbit];
    }
    idxShift >>= sepBit;
    std::cerr << "hi = " << hi << ": idxShift = "
              << utils::as0b(idxShift, hiBits.back()) << "\n";
    pSvs[hi] = B.CreateConstGEP1_64(
      vecTypeX2, ptrSvBeginV, idxShift, "ptr.sv.hi." + std::to_string(hi));

    auto* ampFull = B.CreateLoad(
      vecTypeX2, pSvs[hi], "sv.full.hi." + std::to_string(hi));

    for (unsigned li = 0; li < LK; li++) {
      reAmps[hi * LK + li] = B.CreateShuffleVector(
        ampFull, ArrayRef<int>(reSplitMasks.data() + li * S, S),
        "re." + std::to_string(hi) + "." + std::to_string(li));
      imAmps[hi * LK + li] = B.CreateShuffleVector(
        ampFull, ArrayRef<int>(imSplitMasks.data() + li * S, S),
        "im." + std::to_string(hi) + "." + std::to_string(li));
    }
  }

  SmallVector<Value*> updatedReAmps;
  SmallVector<Value*> updatedImAmps;
  updatedReAmps.resize_for_overwrite(LK);
  updatedImAmps.resize_for_overwrite(LK);
  for (unsigned hi = 0; hi < HK; hi++) {
    // mat-vec mul
    std::memset(updatedReAmps.data(), 0, updatedReAmps.size_in_bytes());
    std::memset(updatedImAmps.data(), 0, updatedImAmps.size_in_bytes());
    for (unsigned li = 0; li < LK; li++) {
      for (unsigned k = 0; k < K; k++) {
        updatedReAmps[li] = internal::genMulAdd(B,
          matrixData[k].reVal, reAmps[k], updatedReAmps[li],
          matrixData[k].reKind, 
          "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedReAmps[li] = internal::genNegMulAdd(B,
          matrixData[k].imVal, imAmps[k], updatedReAmps[li],
          matrixData[k].imKind, 
          "new.re." + std::to_string(hi) + "." + std::to_string(li) + ".");

        updatedImAmps[li] = internal::genMulAdd(B,
          matrixData[k].reVal, imAmps[k], updatedImAmps[li],
          matrixData[k].reKind, 
          "new.im." + std::to_string(hi) + "." + std::to_string(li) + ".");
        updatedImAmps[li] = internal::genMulAdd(B,
          matrixData[k].imVal, reAmps[k], updatedImAmps[li],
          matrixData[k].imKind, 
          "new.im." + std::to_string(hi) + "." + std::to_string(li) + ".");
      }
    }
    
    // store

  }


  loopBodyBB->print(errs());

  // increment counter and return
  auto* taskIdNextV = B.CreateAdd(taskIdV, B.getInt64(1), "taskid.next");
  taskIdV->addIncoming(taskIdNextV, loopBodyBB);
  B.CreateBr(loopBB);

  B.SetInsertPoint(retBB);
  B.CreateRetVoid();

  return func;
}
