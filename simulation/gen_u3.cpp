#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace simulation;

std::string getDefaultU3FuncName(
        const ir::U3Gate& u3, ir::RealTy realTy, ir::AmpFormat ampFormat) {
    std::stringstream ss;
    ss << "u3_"
       << ((realTy == ir::RealTy::Double) ? "f64" : "f32") << "_"
       << ((ampFormat == ir::AmpFormat::Separate) ? "sep" : "alt") << "_"
       << std::setfill('0') << std::setw(8) << std::hex << u3.getID();
    return ss.str();
}


Function* IRGenerator::genU3(const ir::U3Gate& u3,
                             std::string _funcName) {
    const ir::ComplexMatrix2& mat = u3.mat;

    std::string funcName = (_funcName != "") ? _funcName
                         : getDefaultU3FuncName(u3, realTy, ampFormat);

    errs() << "Generating function " << funcName << "\n";

    uint8_t k = u3.k;
    uint64_t _S = 1ULL << vecSizeInBits;
    uint64_t _K = 1ULL << k;
    uint64_t _inner = (1ULL << (k - vecSizeInBits)) - 1;
    uint64_t _outer = ~_inner;
    auto* K = builder.getInt64(_K);
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

    argTy.push_back(builder.getPtrTy()); // ptr to real amp
    argTy.push_back(builder.getPtrTy()); // ptr to imag amp
    argTy.push_back(builder.getInt64Ty()); // idx_start
    argTy.push_back(builder.getInt64Ty()); // idx_end
    argTy.push_back(scalarTyx8); // matrix

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

    auto* arElem = builder.CreateExtractElement(pmat, 0ULL, "arElem");
    auto* brElem = builder.CreateExtractElement(pmat, 1ULL, "brElem");
    auto* crElem = builder.CreateExtractElement(pmat, 2ULL, "crElem");
    auto* drElem = builder.CreateExtractElement(pmat, 3ULL, "drElem");
    auto* aiElem = builder.CreateExtractElement(pmat, 4ULL, "aiElem");
    auto* biElem = builder.CreateExtractElement(pmat, 5ULL, "biElem");
    auto* ciElem = builder.CreateExtractElement(pmat, 6ULL, "ciElem");
    auto* diElem = builder.CreateExtractElement(pmat, 7ULL, "diElem");

    Value* ar = genVectorWithSameElem(scalarTy, _S, arElem, "ar");
    Value* br = genVectorWithSameElem(scalarTy, _S, brElem, "br");
    Value* cr = genVectorWithSameElem(scalarTy, _S, crElem, "cr");
    Value* dr = genVectorWithSameElem(scalarTy, _S, drElem, "dr");
    
    Value* ai = genVectorWithSameElem(scalarTy, _S, aiElem, "ai");
    Value* bi = genVectorWithSameElem(scalarTy, _S, biElem, "bi");
    Value* ci = genVectorWithSameElem(scalarTy, _S, ciElem, "ci");
    Value* di = genVectorWithSameElem(scalarTy, _S, diElem, "di");

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
    auto* ptrAr = builder.CreateGEP(scalarTy, preal, idxA, "ptrAr");
    auto* ptrAi = builder.CreateGEP(scalarTy, pimag, idxA, "ptrAi");
    auto* ptrBr = builder.CreateGEP(scalarTy, preal, idxB, "ptrBr");
    auto* ptrBi = builder.CreateGEP(scalarTy, pimag, idxB, "ptrBi");

    auto* Ar = builder.CreateLoad(vectorTy, ptrAr, "Ar");
    auto* Ai = builder.CreateLoad(vectorTy, ptrAi, "Ai");
    auto* Br = builder.CreateLoad(vectorTy, ptrBr, "Br");
    auto* Bi = builder.CreateLoad(vectorTy, ptrBi, "Bi");

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

    return func;
}

