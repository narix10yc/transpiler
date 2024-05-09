#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace simulation;

std::string getDefaultU3FuncName(const ir::U3Gate& u3, const IRGenerator& gen) {
    std::stringstream ss;
    ss << ((gen.realTy == ir::RealTy::Double) ? "f64" : "f32") << "_"
       << "s" << gen.vecSizeInBits << "_"
       << ((gen.ampFormat == ir::AmpFormat::Separate) ? "sep" : "alt") << "_"
       << u3.getRepr();
    return ss.str();
}

Function* IRGenerator::genU3_Sep(const ir::U3Gate& u3, std::string _funcName) {
    const ir::ComplexMatrix2& mat = u3.mat;
    std::string funcName = (_funcName != "") ? _funcName
                         : getDefaultU3FuncName(u3, *this);

    errs() << "Generating function " << funcName << "\n";

    uint8_t _k = u3.k;
    uint8_t _s = static_cast<uint8_t>(vecSizeInBits);
    uint64_t _S = 1ULL << _s;
    uint64_t _K = 1ULL << _k;
    uint64_t _inner = (1ULL << (_k - _s)) - 1;
    uint64_t _outer = ~_inner;
    auto* K = builder.getInt64(_K);
    auto* inner = builder.getInt64(_inner);
    auto* outer = builder.getInt64(_outer);
    auto* s = builder.getInt64(_s);
    auto* s_add_1 = builder.getInt64(_s + 1);

    Type* scalarTy = (realTy == ir::RealTy::Float) ? builder.getFloatTy()
                                                   : builder.getDoubleTy();
    Type* vectorTy = VectorType::get(scalarTy, _S, false);
    Type* vecTyx2 = VectorType::get(scalarTy, 2 * _S, false);
    Type* scalarTyx8 = VectorType::get(scalarTy, 8, false);

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

    auto* matV = builder.CreateLoad(scalarTyx8, pmat, 0ULL, "mat");

    auto* ar = builder.CreateShuffleVector(matV, std::vector<int>(_S, 0), "ar");
    auto* br = builder.CreateShuffleVector(matV, std::vector<int>(_S, 1), "br");
    auto* cr = builder.CreateShuffleVector(matV, std::vector<int>(_S, 2), "cr");
    auto* dr = builder.CreateShuffleVector(matV, std::vector<int>(_S, 3), "dr");

    auto* ai = builder.CreateShuffleVector(matV, std::vector<int>(_S, 4), "ai");
    auto* bi = builder.CreateShuffleVector(matV, std::vector<int>(_S, 5), "bi");
    auto* ci = builder.CreateShuffleVector(matV, std::vector<int>(_S, 6), "ci");
    auto* di = builder.CreateShuffleVector(matV, std::vector<int>(_S, 7), "di");

    builder.CreateBr(loopBB);

    // loop
    builder.SetInsertPoint(loopBB);
    PHINode* idx = builder.CreatePHI(builder.getInt64Ty(), 2, "idx");
    idx->addIncoming(idx_start, entryBB);
    Value* cond = builder.CreateICmpSLT(idx, idx_end, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    Value *pAr = nullptr, *pAi = nullptr, *pBr = nullptr, *pBi = nullptr;
    Value *pRe = nullptr, *pIm = nullptr;
    std::vector<int> shuffleMaskStore;
    Value *Ar = nullptr, *Ai = nullptr, *Br = nullptr, *Bi = nullptr;
    if (_k >= _s) {
        // idxA = ((idx & outer) << s_add_1) + ((idx & inner) << s)
        auto* idx_and_outer = builder.CreateAnd(idx, outer, "idx_and_outer");
        auto* shifted_outer = builder.CreateShl(idx_and_outer, s_add_1, "shl_outer");
        auto* idx_and_inner = builder.CreateAnd(idx, inner, "idx_and_inner");
        auto* shifted_inner = builder.CreateShl(idx_and_inner, s, "shl_inner");
        auto* idxA = builder.CreateAdd(shifted_outer, shifted_inner, "idxA");
        auto* idxB = builder.CreateAdd(idxA, K, "idxB");

        // A = Ar + i*Ai and B = Br + i*Bi
        pAr = builder.CreateGEP(scalarTy, preal, idxA, "pAr");
        pAi = builder.CreateGEP(scalarTy, pimag, idxA, "pAi");
        pBr = builder.CreateGEP(scalarTy, preal, idxB, "pBr");
        pBi = builder.CreateGEP(scalarTy, pimag, idxB, "pBi");

        Ar = builder.CreateLoad(vectorTy, pAr, "Ar");
        Ai = builder.CreateLoad(vectorTy, pAi, "Ai");
        Br = builder.CreateLoad(vectorTy, pBr, "Br");
        Bi = builder.CreateLoad(vectorTy, pBi, "Bi");
    } else {
        pRe = builder.CreateGEP(vecTyx2, preal, idx, "pRe");
        pIm = builder.CreateGEP(vecTyx2, pimag, idx, "pIm");
        auto* Re = builder.CreateLoad(vecTyx2, pRe, "Re");
        auto* Im = builder.CreateLoad(vecTyx2, pIm, "Im");

        std::vector<int> shuffleMaskLo, shuffleMaskHi;
        for (size_t i = 0; i < 2 * _S; i++) {
            if ((i & _K) == 0) {
                shuffleMaskStore.push_back(shuffleMaskLo.size());
                shuffleMaskLo.push_back(i);
            }
            else {
                shuffleMaskStore.push_back(shuffleMaskHi.size() + _S);
                shuffleMaskHi.push_back(i);
            }
        }
        Ar = builder.CreateShuffleVector(Re, shuffleMaskLo, "Ar");
        Ai = builder.CreateShuffleVector(Im, shuffleMaskLo, "Ai");
        Br = builder.CreateShuffleVector(Re, shuffleMaskHi, "Br");
        Bi = builder.CreateShuffleVector(Im, shuffleMaskHi, "Bi");
    }

    // mat-vec mul (new value should never automatically be 0)
    // newAr = (ar Ar + br Br) - (ai Ai + bi Bi)
    Value* newAr = nullptr;
    newAr = genMulAdd(newAr, ar, Ar, mat.real[0], "arAr", "newAr");
    newAr = genMulAdd(newAr, br, Br, mat.real[1], "brBr", "newAr");
    newAr = genMulSub(newAr, ai, Ai, mat.imag[0], "aiAi", "newAr");
    newAr = genMulSub(newAr, bi, Bi, mat.imag[1], "biBi", "newAr");

    // newAi = ar Ai + ai Ar + br Bi + bi Br
    Value* newAi = nullptr;
    newAi = genMulAdd(newAi, ar, Ai, mat.real[0], "arAi", "newAi");
    newAi = genMulAdd(newAi, ai, Ar, mat.imag[0], "aiAr", "newAi");
    newAi = genMulAdd(newAi, br, Bi, mat.real[1], "brBi", "newAi");
    newAi = genMulAdd(newAi, bi, Br, mat.imag[1], "biBr", "newAi");

    // newBr = (cr Ar + dr Br) - (ci Ai + di Bi)
    Value* newBr = nullptr;
    newBr = genMulAdd(newBr, cr, Ar, mat.real[2], "crAr", "newBr");
    newBr = genMulAdd(newBr, dr, Br, mat.real[3], "drBr", "newBr");
    newBr = genMulSub(newBr, ci, Ai, mat.imag[2], "ciAi", "newBr");
    newBr = genMulSub(newBr, di, Bi, mat.imag[3], "diBi", "newBr");

    // newBi = cr Ai + ci Ar + di Br + dr Bi
    Value* newBi = nullptr;
    newBi = genMulAdd(newBi, cr, Ai, mat.real[2], "crAi", "newBi");
    newBi = genMulAdd(newBi, ci, Ar, mat.imag[2], "ciAr", "newBi");
    newBi = genMulAdd(newBi, di, Br, mat.imag[3], "diBr", "newBi");
    newBi = genMulAdd(newBi, dr, Bi, mat.real[3], "drBi", "newBi");

    // store back 
    if (_k >= _s) {
        builder.CreateStore(newAr, pAr);
        builder.CreateStore(newAi, pAi);
        builder.CreateStore(newBr, pBr);
        builder.CreateStore(newBi, pBi);
    } else {
        auto* newRe = builder.CreateShuffleVector(newAr, newBr, shuffleMaskStore, "newRe");
        auto* newIm = builder.CreateShuffleVector(newAi, newBi, shuffleMaskStore, "newIm");
        builder.CreateStore(newRe, pRe);
        builder.CreateStore(newIm, pIm);
    }

    auto* idx_next = builder.CreateAdd(idx, builder.getInt64(1), "idx_next");
    idx->addIncoming(idx_next, loopBodyBB);
    builder.CreateBr(loopBB);

    // return 
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}

Function* IRGenerator::genU3_Alt(const ir::U3Gate& u3, std::string _funcName) {
    const ir::ComplexMatrix2& mat = u3.mat;
    std::string funcName = (_funcName != "") ? _funcName
                         : getDefaultU3FuncName(u3, *this);

    errs() << "Generating function " << funcName << "\n";

    uint8_t k = u3.k;
    uint64_t _S = 1ULL << vecSizeInBits;
    uint64_t _Kx2 = 1ULL << (k + 1);
    uint64_t _inner = (1ULL << (k - vecSizeInBits + 1)) - 1;
    uint64_t _outer = ~_inner;
    auto* Kx2 = builder.getInt64(_Kx2);
    auto* inner = builder.getInt64(_inner);
    auto* outer = builder.getInt64(_outer);
    auto* s = builder.getInt64(vecSizeInBits);
    auto* s_add_1 = builder.getInt64(vecSizeInBits + 1);

    Type* scalarTy = (realTy == ir::RealTy::Float) ? builder.getFloatTy()
                                                   : builder.getDoubleTy();
    Type* vectorTy = VectorType::get(scalarTy, _S, false);
    Type* scalarTyx8 = VectorType::get(scalarTy, 8, false);

    // create function
    Type* retTy = builder.getVoidTy();
    SmallVector<Type*> argTy;

    argTy.push_back(builder.getPtrTy()); // ptr to sv
    argTy.push_back(builder.getInt64Ty()); // idx_start
    argTy.push_back(builder.getInt64Ty()); // idx_end
    argTy.push_back(builder.getPtrTy()); // matrix

    FunctionType* funcTy = FunctionType::get(retTy, argTy, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, mod.get());

    auto* psv = func->getArg(0);
    auto* idx_start = func->getArg(1);
    auto* idx_end = func->getArg(2);
    auto* pmat = func->getArg(3);

    SmallVector<StringRef> argNames { "psv", "idx_start", "idx_end", "pmat"};

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

    auto* matV = builder.CreateLoad(scalarTyx8, pmat, 0ULL, "mat");

    auto* ar = builder.CreateShuffleVector(matV, std::vector<int>(_S, 0), "ar");
    auto* br = builder.CreateShuffleVector(matV, std::vector<int>(_S, 1), "br");
    auto* cr = builder.CreateShuffleVector(matV, std::vector<int>(_S, 2), "cr");
    auto* dr = builder.CreateShuffleVector(matV, std::vector<int>(_S, 3), "dr");

    auto* ai = builder.CreateShuffleVector(matV, std::vector<int>(_S, 4), "ai");
    auto* bi = builder.CreateShuffleVector(matV, std::vector<int>(_S, 5), "bi");
    auto* ci = builder.CreateShuffleVector(matV, std::vector<int>(_S, 6), "ci");
    auto* di = builder.CreateShuffleVector(matV, std::vector<int>(_S, 7), "di");

    Constant* negV;
    if (realTy == ir::RealTy::Double) {
        std::vector<double> _negV;
        for (size_t i = 0; i < _S; i++)
            _negV.push_back((i % 2 == 0) ? 1 : -1);
        negV = ConstantDataVector::get(llvmContext, _negV);
    } else {
        std::vector<float> _negV;
        for (size_t i = 0; i < _S; i++)
            _negV.push_back((i % 2 == 0) ? 1 : -1);
        negV = ConstantDataVector::get(llvmContext, _negV);
    }

    auto* ai_n = builder.CreateFMul(ai, negV, "ai_n");
    auto* bi_n = builder.CreateFMul(bi, negV, "bi_n");
    auto* ci_n = builder.CreateFMul(ci, negV, "ci_n");
    auto* di_n = builder.CreateFMul(di, negV, "di_n");

    std::vector<int> _shuffleMask;
    for (size_t i = 0; i < _S; i++)
        _shuffleMask.push_back(i ^ 1);

    builder.CreateBr(loopBB);

    // loop
    builder.SetInsertPoint(loopBB);
    PHINode* idx = builder.CreatePHI(builder.getInt64Ty(), 2, "idx");
    idx->addIncoming(idx_start, entryBB);
    Value* cond = builder.CreateICmpSLT(idx, idx_end, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    // idxA = ((idx & outer) << s_add_2) + ((idx & inner) << s_add_1)
    auto* idx_and_outer = builder.CreateAnd(idx, outer, "idx_and_outer");
    auto* shifted_outer = builder.CreateShl(idx_and_outer, s_add_1, "shl_outer");
    auto* idx_and_inner = builder.CreateAnd(idx, inner, "idx_and_inner");
    auto* shifted_inner = builder.CreateShl(idx_and_inner, s, "shl_inner");
    auto* idxA = builder.CreateAdd(shifted_outer, shifted_inner, "alpha");
    auto* idxB = builder.CreateAdd(idxA, Kx2, "beta");


    // Lo = sv[idxA], Hi = sv[idxB]
    auto* ptrLo = builder.CreateGEP(scalarTy, psv, idxA, "ptrLo");
    auto* ptrHi = builder.CreateGEP(scalarTy, psv, idxB, "ptrHi");

    auto* Lo = builder.CreateLoad(vectorTy, ptrLo, "Lo");
    auto* Hi = builder.CreateLoad(vectorTy, ptrHi, "Hi");

    // newLo = (ar Lo + br Hi) + i(ai Lo + bi Hi)
    Value* LoRe = nullptr;
    LoRe = genMulAdd(LoRe, ar, Lo, mat.real[0], "arLo", "LoRe");
    LoRe = genMulAdd(LoRe, br, Hi, mat.real[1], "brHi", "LoRe");

    Value* LoIm = nullptr;
    LoIm = genMulAdd(LoIm, ai_n, Lo, mat.imag[0], "aiLo", "LoIm_s");
    LoIm = genMulAdd(LoIm, bi_n, Hi, mat.imag[1], "biHi", "LoIm_s");
    if (LoIm != nullptr)
        LoIm = builder.CreateShuffleVector(LoIm, _shuffleMask, "LoIm");

    Value* newLo = nullptr;
    if (LoRe != nullptr && LoIm != nullptr)
        newLo = builder.CreateFAdd(LoRe, LoIm, "newLo");
    else if (LoRe != nullptr)
        newLo = LoRe;
    else if (LoIm != nullptr)
        newLo = LoIm;
    else { /* should never happen */ }

    // newHi = (cr Lo + dr Hi) + i(ci Lo + di Hi)
    Value* HiRe = nullptr;
    HiRe = genMulAdd(HiRe, cr, Lo, mat.real[2], "crLo", "HiRe");
    HiRe = genMulAdd(HiRe, dr, Hi, mat.real[3], "drHi", "HiRe");

    Value* HiIm = nullptr;
    HiIm = genMulAdd(HiIm, ci_n, Lo, mat.imag[2], "ciLo", "HiIm_s");
    HiIm = genMulAdd(HiIm, di_n, Hi, mat.imag[3], "diHi", "HiIm_s");
    if (HiIm != nullptr)
        HiIm = builder.CreateShuffleVector(HiIm, _shuffleMask, "HiIm");

    Value* newHi = nullptr;
    if (HiRe != nullptr && HiIm != nullptr) 
        newHi = builder.CreateFAdd(HiRe, HiIm, "newHi");
    else if (HiRe != nullptr)
        newHi = HiRe;
    else if (HiIm != nullptr)
        newHi = HiIm;
    else { /* should never happen */ }

    // store back 
    builder.CreateStore(newLo, ptrLo);
    builder.CreateStore(newHi, ptrHi);

    auto* idx_next = builder.CreateAdd(idx, builder.getInt64(1), "idx_next");
    idx->addIncoming(idx_next, loopBodyBB);
    builder.CreateBr(loopBB);

    // return 
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}