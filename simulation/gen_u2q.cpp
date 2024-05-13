#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace simulation;

std::string getDefaultU2qFuncName(const ir::U2qGate& u2q, const IRGenerator& gen) {
    std::stringstream ss;
    ss << ((gen.realTy == ir::RealTy::Double) ? "f64" : "f32") << "_"
       << "s" << gen.vecSizeInBits << "_"
       << ((gen.ampFormat == ir::AmpFormat::Separate) ? "sep" : "alt") << "_"
       << u2q.getRepr();
    return ss.str();
}

Function* IRGenerator::genU2q(const ir::U2qGate& u2q, std::string _funcName) {
    const ir::ComplexMatrix4& mat = u2q.mat;
    std::string funcName = (_funcName != "") ? _funcName
                         : getDefaultU2qFuncName(u2q, *this);

    errs() << "Generating function " << funcName << "\n";

    // convention: l is the less significant qubit
    uint8_t l = u2q.qSmall;
    uint8_t k = u2q.qLarge;
    uint8_t s = static_cast<uint8_t>(vecSizeInBits);
    uint64_t L = 1ULL << l;
    uint64_t S = 1ULL << s;
    uint64_t K = 1ULL << k;
    uint64_t leftMask = ~((1 << (k-s-1)) - 1);
    uint64_t middleMask = ((1 << (k-l-1)) - 1) << (l-s);
    uint64_t rightMask = (1 << (l-s)) - 1;

    auto* KVal = builder.getInt64(K);
    auto* LVal = builder.getInt64(L);
    auto* KorLVal = builder.getInt64(K | L);
    auto* leftMaskVal = builder.getInt64(leftMask);
    auto* middleMaskVal = builder.getInt64(middleMask);
    auto* rightMaskVal = builder.getInt64(rightMask);
    auto* sVal = builder.getInt64(s);
    auto* s_add_1_Val = builder.getInt64(s + 1);
    auto* s_add_2_Val = builder.getInt64(s + 2);

    Type* scalarTy = (realTy == ir::RealTy::Float) ? builder.getFloatTy()
                                                   : builder.getDoubleTy();
    Type* vecSTy = VectorType::get(scalarTy, S, false);
    Type* vec2STy = VectorType::get(scalarTy, 2 * S, false);
    Type* vec4STy = VectorType::get(scalarTy, 4 * S, false);
    Type* vec32Ty = VectorType::get(scalarTy, 32, false);

    // create function
    Type* retTy = builder.getVoidTy();
    SmallVector<Type*> argTy;

    argTy.push_back(builder.getPtrTy()); // ptr to real amp
    argTy.push_back(builder.getPtrTy()); // ptr to imag amp
    argTy.push_back(builder.getInt64Ty()); // idx_start
    argTy.push_back(builder.getInt64Ty()); // idx_end
    argTy.push_back(builder.getPtrTy()); // ptr to matrix

    FunctionType* funcTy = FunctionType::get(retTy, argTy, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, mod.get());

    auto* preal = func->getArg(0);
    auto* pimag = func->getArg(1);
    auto* idx_start = func->getArg(2);
    auto* idx_end = func->getArg(3);
    auto* pmat = func->getArg(4);

    SmallVector<StringRef> argNames
        { "preal", "pimag", "idx_start", "idx_end", "pmat"};

    // set arg names
    size_t i = 0;
    for (auto& arg : func->args())
        arg.setName(argNames[i++]);

    // init basic blocks
    BasicBlock* entryBB = BasicBlock::Create(llvmContext, "entry", func);
    BasicBlock* loopBB = BasicBlock::Create(llvmContext, "loop", func);
    BasicBlock* loopBodyBB = BasicBlock::Create(llvmContext, "loopBody", func);
    BasicBlock* retBB = BasicBlock::Create(llvmContext, "ret", func);

    builder.SetInsertPoint(entryBB);
    // load matrix
    auto* matV = builder.CreateLoad(vec32Ty, pmat, 0ULL, "mat");
    Value *mRe[16], *mIm[16];
    for (size_t i = 0; i < 16; i++) {
        mRe[i] = builder.CreateShuffleVector(
            matV, std::vector<int>(S, i), "mRe" + std::to_string(i));
    }
    for (size_t i = 0; i < 16; i++) {
        mIm[i] = builder.CreateShuffleVector(
            matV, std::vector<int>(S, i+16), "mIm" + std::to_string(i));
    }

    builder.CreateBr(loopBB);

    // loop
    builder.SetInsertPoint(loopBB);
    PHINode* idx = builder.CreatePHI(builder.getInt64Ty(), 2, "idx");
    idx->addIncoming(idx_start, entryBB);
    Value* cond = builder.CreateICmpSLT(idx, idx_end, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    Value *pRe[4], *pIm[4], *Re[4], *Im[4];
    Value *pReal = nullptr, *pImag = nullptr;
    std::vector<int> shuffleMaskStore;
    if (l >= s) {
        // k > l >= s
        auto* leftVal = builder.CreateAnd(idx, leftMaskVal, "left_tmp");
        leftVal = builder.CreateShl(leftVal, s_add_2_Val, "left");
        auto* middleVal = builder.CreateAnd(idx, middleMaskVal, "middle_tmp");
        middleVal = builder.CreateShl(middleVal, s_add_1_Val, "middle");
        auto rightVal = builder.CreateAnd(idx, rightMaskVal, "right_tmp");
        rightVal = builder.CreateShl(rightVal, sVal, "right");
        auto* idx0 = builder.CreateOr(leftVal, middleVal, "idx0_tmp");
        idx0 = builder.CreateOr(idx0, rightVal, "idx0");
        auto* idx1 = builder.CreateOr(idx0, LVal, "idx1");
        auto* idx2 = builder.CreateOr(idx0, KVal, "idx2");
        auto* idx3 = builder.CreateOr(idx0, KorLVal, "idx3");

        pRe[0] = builder.CreateGEP(scalarTy, preal, idx0, "pRe0");
        pRe[1] = builder.CreateGEP(scalarTy, preal, idx1, "pRe1");
        pRe[2] = builder.CreateGEP(scalarTy, preal, idx2, "pRe2");
        pRe[3] = builder.CreateGEP(scalarTy, preal, idx3, "pRe3");
        pIm[0] = builder.CreateGEP(scalarTy, pimag, idx0, "pIm0");
        pIm[1] = builder.CreateGEP(scalarTy, pimag, idx1, "pIm1");
        pIm[2] = builder.CreateGEP(scalarTy, pimag, idx2, "pIm2");
        pIm[3] = builder.CreateGEP(scalarTy, pimag, idx3, "pIm3");

        Re[0] = builder.CreateLoad(vecSTy, pRe[0], "Re0");
        Re[1] = builder.CreateLoad(vecSTy, pRe[1], "Re1");
        Re[2] = builder.CreateLoad(vecSTy, pRe[2], "Re2");
        Re[3] = builder.CreateLoad(vecSTy, pRe[3], "Re3");
        Im[0] = builder.CreateLoad(vecSTy, pIm[0], "Im0");
        Im[1] = builder.CreateLoad(vecSTy, pIm[1], "Im1");
        Im[2] = builder.CreateLoad(vecSTy, pIm[2], "Im2");
        Im[3] = builder.CreateLoad(vecSTy, pIm[3], "Im3");
    } else if (s > k) {
        // s > k > l
        pReal = builder.CreateGEP(vec4STy, preal, idx, "pReal");
        pImag = builder.CreateGEP(vec4STy, pimag, idx, "pImag");
        auto* Real = builder.CreateLoad(vec4STy, pReal, "Real");
        auto* Imag = builder.CreateLoad(vec4STy, pImag, "Imag");

        std::vector<int> shuffleMasks[4];
        for (size_t i = 0; i < 4 * S; i++) {
            size_t position = ((i >> l) & 1U) + ((i >> (k-1)) & 2U);
            shuffleMaskStore.push_back(shuffleMasks[position].size() + position * S);
            shuffleMasks[position].push_back(i);
        }
        for (size_t i = 0; i < 4; i++)
            Re[i] = builder.CreateShuffleVector(Real, shuffleMasks[i], "Re" + std::to_string(i));
        for (size_t i = 0; i < 4; i++)
            Im[i] = builder.CreateShuffleVector(Imag, shuffleMasks[i], "Im" + std::to_string(i));  
    } else {
        // k >= s > l
        /* TODO */
    }

    // mat-vec multiplication
    Value *newRe[4] = {nullptr}, *newIm[4] = {nullptr};
    for (size_t i = 0; i < 4; i++) {
        Value *newRe0 = nullptr, *newRe1 = nullptr;
        size_t i0 = 4*i + 0, i1 = 4*i + 1, i2 = 4*i + 2, i3 = 4*i + 3;
        std::string newReName = "newRe" + std::to_string(i) + "_";

        newRe0 = genMulAdd(newRe0, mRe[i0], Re[0], mat.real[i0], "", newReName);
        newRe0 = genMulAdd(newRe0, mRe[i1], Re[1], mat.real[i1], "", newReName);
        newRe0 = genMulAdd(newRe0, mRe[i2], Re[2], mat.real[i2], "", newReName);
        newRe0 = genMulAdd(newRe0, mRe[i3], Re[3], mat.real[i3], "", newReName);

        newRe1 = genMulAdd(newRe1, mIm[i0], Im[0], mat.imag[i0], "", newReName);
        newRe1 = genMulAdd(newRe1, mIm[i1], Im[1], mat.imag[i1], "", newReName);
        newRe1 = genMulAdd(newRe1, mIm[i2], Im[2], mat.imag[i2], "", newReName);
        newRe1 = genMulAdd(newRe1, mIm[i3], Im[3], mat.imag[i3], "", newReName);
        
        if (newRe0 != nullptr && newRe1 != nullptr)
            newRe[i] = builder.CreateFSub(newRe0, newRe1, "newRe" + std::to_string(i));
        else if (newRe0 == nullptr)
            newRe[i] = builder.CreateFNeg(newRe1, "newRe" + std::to_string(i));
        else if (newRe1 == nullptr)
            newRe[i] = newRe0;
        else // should never happen
            newRe[i] = nullptr;
    }

    for (size_t i = 0; i < 4; i++) {
        size_t i0 = 4*i + 0, i1 = 4*i + 1, i2 = 4*i + 2, i3 = 4*i + 3;
        std::string newImName = "newIm" + std::to_string(i) + "_";

        newIm[i] = genMulAdd(newIm[i], mRe[i0], Im[0], mat.real[i0], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mRe[i1], Im[1], mat.real[i1], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mRe[i2], Im[2], mat.real[i2], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mRe[i3], Im[3], mat.real[i3], "", newImName);

        newIm[i] = genMulAdd(newIm[i], mIm[i0], Re[0], mat.imag[i0], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mIm[i1], Re[1], mat.imag[i1], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mIm[i2], Re[2], mat.imag[i2], "", newImName);
        newIm[i] = genMulAdd(newIm[i], mIm[i3], Re[3], mat.imag[i3], "", newImName);
    }
    
    // store back
    if (l >= s) {
        for (size_t i = 0; i < 4; i++)
            builder.CreateStore(newRe[i], pRe[i]);
        for (size_t i = 0; i < 4; i++)
            builder.CreateStore(newIm[i], pIm[i]);
    } else if (s > k) {
        std::vector<int> shuffleMaskCombine;
        for (size_t i = 0; i < 2 * S; i++)
            shuffleMaskCombine.push_back(i);
        
        auto* vecRe0 = builder.CreateShuffleVector(newRe[0], newRe[1], shuffleMaskCombine, "vecRe0");
        auto* vecRe1 = builder.CreateShuffleVector(newRe[2], newRe[3], shuffleMaskCombine, "vecRe1");
        auto* newReal = builder.CreateShuffleVector(vecRe0, vecRe1, shuffleMaskStore, "newReal");
        builder.CreateStore(newReal, pReal);

        auto* vecIm0 = builder.CreateShuffleVector(newIm[0], newIm[1], shuffleMaskCombine, "vecIm0");
        auto* vecIm1 = builder.CreateShuffleVector(newIm[2], newIm[3], shuffleMaskCombine, "vecIm1");
        auto* newImag = builder.CreateShuffleVector(vecIm0, vecIm1, shuffleMaskStore, "newImag");
        builder.CreateStore(newImag, pImag);
    }
    else {
        /* TODO */
    }

    auto* idx_next = builder.CreateAdd(idx, builder.getInt64(1), "idx_next");
    idx->addIncoming(idx_next, loopBodyBB);
    builder.CreateBr(loopBB);

    // return 
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}
