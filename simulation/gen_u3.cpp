#include "simulation/irGen.h"
#include <iostream>

using namespace llvm;
using namespace simulation;

Value* IRGenerator::genMulAddOrMulSub(Value* aa, bool add, Value* bb, Value* cc, 
        int bbFlag, const Twine& bbccName, const Twine& aaName) {
    if (bbFlag == 0) 
        return aa;
    
    if (bbFlag == 1) {
        if (aa == nullptr) {
            if (add) return cc;
            return builder.CreateFNeg(cc, aaName);
        }
        if (add) return builder.CreateFAdd(aa, cc, aaName);
        return builder.CreateFSub(aa, cc, aaName);
    }

    if (bbFlag == -1) {
        if (aa == nullptr) {
            if (add) return builder.CreateFNeg(cc, aaName);
            return cc;
        }
        if (add) return builder.CreateFSub(aa, cc, aaName);
        return builder.CreateFAdd(aa, cc, aaName);
    }

    auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
    if (aa == nullptr) {
        if (add) return bbcc;
        return builder.CreateFNeg(bbcc, aaName);
    }
    if (add)
        return builder.CreateFAdd(aa, bbcc, aaName);
    return builder.CreateFSub(aa, bbcc, aaName);
}


Value* IRGenerator::genMulAdd(Value* aa, Value* bb, Value* cc, 
                              int bbFlag, const Twine& bbccName,
                              const Twine& aaName) {
    if (bbFlag == 0) 
        return aa;
    
    // new_aa = aa + cc
    if (bbFlag == 1) {
        if (aa == nullptr)
            return cc;
        return builder.CreateFAdd(aa, cc, aaName);
    }

    // new_aa = aa - cc
    if (bbFlag == -1) {
        if (aa == nullptr)
            return builder.CreateFNeg(cc, aaName);
        return builder.CreateFSub(aa, cc, aaName);
    }

    // new_aa = aa + bb * cc
    auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
    if (aa == nullptr)
        return bbcc;
    return builder.CreateFAdd(aa, bbcc, aaName);
}

Value* IRGenerator::genMulSub(Value* aa, Value* bb, Value* cc, 
                              int bbFlag, const Twine& bbccName,
                              const Twine& aaName) {
    if (bbFlag == 0) 
        return aa;
    
    // new_aa = aa - cc
    if (bbFlag == 1) {
        if (aa == nullptr)
            return builder.CreateFNeg(cc, aaName);
        return builder.CreateFSub(aa, cc, aaName);
    }

    // new_aa = aa + cc
    if (bbFlag == -1) {
        if (aa == nullptr)
            return cc;
        return builder.CreateFAdd(aa, cc, aaName);
    }

    // new_aa = aa - bb * cc
    auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
    if (aa == nullptr)
        return builder.CreateFNeg(bbcc, aaName);
    return builder.CreateFSub(aa, bbcc, aaName);
}


void IRGenerator::genU3(const U3Gate& u3,
                        const llvm::StringRef funcName, 
                        RealTy realType) {
    const OptionalComplexMat2x2& mat = u3.mat;

    errs() << "Generating function " << funcName << "\n";

    uint8_t k = u3.qubit;
    uint64_t _S = 1ULL << vecSizeInBits;
    uint64_t _K = 1ULL << k;
    uint64_t _inner = (1ULL << (k - vecSizeInBits)) - 1;
    uint64_t _outer = ~_inner;
    auto* K = builder.getInt64(_K);
    auto* inner = builder.getInt64(_inner);
    auto* outer = builder.getInt64(_outer);
    auto* s = builder.getInt64(vecSizeInBits);
    auto* s_add_1 = builder.getInt64(vecSizeInBits + 1);

    Type* realTy = (realType == RealTy::Float) ? builder.getFloatTy()
                                               : builder.getDoubleTy();
    Type* vectorTy = VectorType::get(realTy, _S, false);
    Type* realTyx8 = VectorType::get(realTy, 8, false);

    // create function
    Type* retTy = builder.getVoidTy();
    SmallVector<Type*> argTy;

    argTy.push_back(builder.getPtrTy()); // ptr to real amp
    argTy.push_back(builder.getPtrTy()); // ptr to imag amp
    argTy.push_back(builder.getInt64Ty()); // idx_start
    argTy.push_back(builder.getInt64Ty()); // idx_end
    argTy.push_back(realTyx8); // matrix

    FunctionType* funcTy = FunctionType::get(retTy, argTy, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, mod.get());

    auto* preal = func->getArg(0);
    auto* pimag = func->getArg(1);
    auto* idx_start = func->getArg(2);
    auto* idx_end = func->getArg(3);
    Value* pmat = func->getArg(4);

    SmallVector<StringRef> argNames
        { "preal", "pimag", "idx_start", "idx_end", "mat"};

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

    // load matrix to vector reg
    // Special optimization only applies when +1, -1, or 0
    auto getFlag = [](std::optional<double> v) -> int {
        if (!v.has_value()) return 2; 
        if (v.value() == 1) return 1;
        if (v.value() == -1) return -1;
        if (v.value() == 0) return 0;
        return 2;
    };

    int arFlag = getFlag(mat.ar);
    int brFlag = getFlag(mat.br);
    int crFlag = getFlag(mat.cr);
    int drFlag = getFlag(mat.dr);
    int aiFlag = getFlag(mat.ai);
    int biFlag = getFlag(mat.bi);
    int ciFlag = getFlag(mat.ci);
    int diFlag = getFlag(mat.di);

    auto* arElem = builder.CreateExtractElement(pmat, (uint64_t)0, "arElem");
    auto* brElem = builder.CreateExtractElement(pmat, 1, "brElem");
    auto* crElem = builder.CreateExtractElement(pmat, 2, "crElem");
    auto* drElem = builder.CreateExtractElement(pmat, 3, "drElem");
    auto* aiElem = builder.CreateExtractElement(pmat, 4, "aiElem");
    auto* biElem = builder.CreateExtractElement(pmat, 5, "biElem");
    auto* ciElem = builder.CreateExtractElement(pmat, 6, "ciElem");
    auto* diElem = builder.CreateExtractElement(pmat, 7, "diElem");

    Value* ar = getVectorWithSameElem(realTy, _S, arElem, "ar");
    Value* br = getVectorWithSameElem(realTy, _S, brElem, "br");
    Value* cr = getVectorWithSameElem(realTy, _S, crElem, "cr");
    Value* dr = getVectorWithSameElem(realTy, _S, drElem, "dr");
    
    Value* ai = getVectorWithSameElem(realTy, _S, aiElem, "ai");
    Value* bi = getVectorWithSameElem(realTy, _S, biElem, "bi");
    Value* ci = getVectorWithSameElem(realTy, _S, ciElem, "ci");
    Value* di = getVectorWithSameElem(realTy, _S, diElem, "di");

    builder.CreateBr(loopBB);

    // loop
    builder.SetInsertPoint(loopBB);
    PHINode* idx = builder.CreatePHI(builder.getInt64Ty(), 2, "idx");
    idx->addIncoming(idx_start, entryBB);
    Value* cond = builder.CreateICmpSLT(idx, idx_end, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    // idxA = ((idx & outer) << s_add_1) + ((idx & inner) << s)
    auto* idx_and_outer = builder.CreateAnd(idx, outer, "idx_and_outer");
    auto* shifted_outer = builder.CreateShl(idx_and_outer, s_add_1, "shl_outer");
    auto* idx_and_inner = builder.CreateAnd(idx, inner, "idx_and_inner");
    auto* shifted_inner = builder.CreateShl(idx_and_inner, s, "shl_inner");
    auto* idxA = builder.CreateAdd(shifted_outer, shifted_inner, "alpha");
    auto* idxB = builder.CreateAdd(idxA, K, "beta");

    // A = Ar + i*Ai and B = Br + i*Bi
    auto* ptrAr = builder.CreateGEP(realTy, preal, idxA, "ptrAr");
    auto* ptrAi = builder.CreateGEP(realTy, pimag, idxA, "ptrAi");
    auto* ptrBr = builder.CreateGEP(realTy, preal, idxB, "ptrBr");
    auto* ptrBi = builder.CreateGEP(realTy, pimag, idxB, "ptrBi");

    auto* Ar = builder.CreateLoad(vectorTy, ptrAr, "Ar");
    auto* Ai = builder.CreateLoad(vectorTy, ptrAi, "Ai");
    auto* Br = builder.CreateLoad(vectorTy, ptrBr, "Br");
    auto* Bi = builder.CreateLoad(vectorTy, ptrBi, "Bi");

    // mat-vec mul (new value should never automatically be 0)
    // newAr = (ar Ar + br Br) - (ai Ai + bi Bi)
    Value* newAr = nullptr;
    newAr = genMulAddOrMulSub(newAr, true, ar, Ar, arFlag, "arAr", "newAr");
    newAr = genMulAddOrMulSub(newAr, true, br, Br, brFlag, "brBr", "newAr");
    newAr = genMulAddOrMulSub(newAr, false, ai, Ai, aiFlag, "aiAi", "newAr");
    newAr = genMulAddOrMulSub(newAr, false, bi, Bi, biFlag, "biBi", "newAr");

    // newAi = ar Ai + ai Ar + br Bi + bi Br
    Value* newAi = nullptr;
    newAi = genMulAddOrMulSub(newAi, true, ar, Ai, arFlag, "arAi", "newAi");
    newAi = genMulAddOrMulSub(newAi, true, ai, Ar, aiFlag, "aiAr", "newAi");
    newAi = genMulAddOrMulSub(newAi, true, br, Bi, brFlag, "brBi", "newAi");
    newAi = genMulAddOrMulSub(newAi, true, bi, Br, biFlag, "biBr", "newAi");

    // newBr = (cr Ar + dr Br) - (ci Ai + di Bi)
    Value* newBr = nullptr;
    newBr = genMulAddOrMulSub(newBr, true, cr, Ar, crFlag, "crAr", "newBr");
    newBr = genMulAddOrMulSub(newBr, true, dr, Br, drFlag, "drBr", "newBr");
    newBr = genMulAddOrMulSub(newBr, false, ci, Ai, ciFlag, "ciAi", "newBr");
    newBr = genMulAddOrMulSub(newBr, false, di, Bi, diFlag, "diBi", "newBr");

    // newBi = cr Ai + ci Ar + di Br + dr Bi
    Value* newBi = nullptr;
    newBi = genMulAddOrMulSub(newBi, true, cr, Ai, crFlag, "crAi", "newBi");
    newBi = genMulAddOrMulSub(newBi, true, ci, Ar, ciFlag, "ciAr", "newBi");
    newBi = genMulAddOrMulSub(newBi, true, di, Br, diFlag, "diBr", "newBi");
    newBi = genMulAddOrMulSub(newBi, true, dr, Bi, drFlag, "drBi", "newBi");

    // store back 
    builder.CreateStore(newAr, ptrAr);
    builder.CreateStore(newAi, ptrAi);
    builder.CreateStore(newBr, ptrBr);
    builder.CreateStore(newBi, ptrBi);

    auto* idx_next = builder.CreateAdd(idx, builder.getInt64(1), "idx_next");
    idx->addIncoming(idx_next, loopBodyBB);
    builder.CreateBr(loopBB);

    // return 
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();
}

void IRGenerator::genU2q(const U2qGate& u2q,
            const llvm::StringRef funcName,
            RealTy realType) {
    auto mat = u2q.mat;
}
