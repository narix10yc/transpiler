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


void IRGenerator::genU3(const int64_t k, 
        const llvm::StringRef funcName, 
        const RealTy realType,
        std::optional<double> _theta, 
        std::optional<double> _phi, 
        std::optional<double> _lambd, 
        double thres) {
    auto mat = OptionalComplexMat2x2::FromAngles(_theta, _phi, _lambd, thres);

    errs() << mat.ar.has_value() << " " <<  mat.br.has_value() << " "
        << mat.cr.has_value() << " " << mat.dr.has_value() << " "
        << mat.bi.has_value() << " " << mat.ci.has_value() << " "
        << mat.di.has_value() << "\n";

    errs() << "Generating function " << funcName << "\n";

    int64_t _S = 1 << vecSizeInBits;
    int64_t _K = 1 << k;
    int64_t _inner = (1 << (k - vecSizeInBits)) - 1;
    int64_t _outer = static_cast<int64_t>(-1) - _inner;
    auto* K = builder.getInt64(_K);
    auto* inner = builder.getInt64(_inner);
    auto* outer = builder.getInt64(_outer);
    auto* s = builder.getInt64(vecSizeInBits);
    auto* s_add_1 = builder.getInt64(vecSizeInBits + 1);

    Type* realTy = (realType == RealTy::Float) ? builder.getFloatTy() : builder.getDoubleTy();
    Type* vectorTy = VectorType::get(realTy, _S, false);

    // create function
    Type* retTy = builder.getVoidTy();
    SmallVector<Type*> argTy;

    argTy.push_back(builder.getPtrTy()); // ptr to real amp
    argTy.push_back(builder.getPtrTy()); // ptr to imag amp
    argTy.push_back(builder.getInt64Ty()); // idx_start
    argTy.push_back(builder.getInt64Ty()); // idx_end
    if (!_theta.has_value()) 
        argTy.push_back(realTy); // theta
    if (!_phi.has_value()) 
        argTy.push_back(realTy); // phi
    if (!_lambd.has_value()) 
        argTy.push_back(realTy); // lambd

    FunctionType* funcTy = FunctionType::get(retTy, argTy, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, mod.get());

    SmallVector<StringRef> argNames { "preal", "pimag", "idx_start", "idx_end" };
    if (!_theta.has_value()) 
        argNames.push_back("theta");
    if (!_phi.has_value()) 
        argNames.push_back("phi");
    if (!_lambd.has_value()) 
        argNames.push_back("lambda");

    auto* preal = func->getArg(0);
    auto* pimag = func->getArg(1);
    auto* idx_start = func->getArg(2);
    auto* idx_end = func->getArg(3);
    Value *theta = nullptr, *phi = nullptr, *lambd = nullptr;

    size_t i = 0;
    for (auto& arg : func->args()) {
        arg.setName(argNames[i]);
        if (argNames[i] == "theta")
            theta = func->getArg(i);
        if (argNames[i] == "phi") 
            phi = func->getArg(i); 
        if (argNames[i] == "lambda")
            lambd = func->getArg(i);
        ++i;
    }

    assert(_theta.has_value() ^ (theta != nullptr));
    assert(_phi.has_value() ^ (phi != nullptr));
    assert(_lambd.has_value() ^ (lambd != nullptr));

    // init basic blocks
    BasicBlock* entryBB = BasicBlock::Create(llvmContext, "entry", func);
    BasicBlock* loopBB = BasicBlock::Create(llvmContext, "loop", func);
    BasicBlock* loopBodyBB = BasicBlock::Create(llvmContext, "loopBody", func);
    BasicBlock* retBB = BasicBlock::Create(llvmContext, "ret", func);

    builder.SetInsertPoint(entryBB);
    Value *ctheta, *stheta, *cphi, *sphi, *clambd, *slambd, *cphi_lambd, *sphi_lambd;
    // All theta, if used, will present in the form of theta/2
    if (theta) {
        theta = builder.CreateFMul(theta, ConstantFP::get(realTy, 0.5), "theta");
        ctheta = builder.CreateUnaryIntrinsic(Intrinsic::cos, theta, nullptr, "cos_theta");
        stheta = builder.CreateUnaryIntrinsic(Intrinsic::sin, theta, nullptr, "sin_theta");
    } else {
        ctheta = ConstantFP::get(realTy, cos(_theta.value() * 0.5));
        stheta = ConstantFP::get(realTy, sin(_theta.value() * 0.5));
    }

    if (phi) {
        cphi = builder.CreateUnaryIntrinsic(Intrinsic::cos, phi, nullptr, "cos_phi");
        sphi = builder.CreateUnaryIntrinsic(Intrinsic::sin, phi, nullptr, "sin_phi");
    } else {
        cphi = ConstantFP::get(realTy, cos(_phi.value()));
        sphi = ConstantFP::get(realTy, sin(_phi.value()));
    }
    
    if (lambd) {
        clambd = builder.CreateUnaryIntrinsic(Intrinsic::cos, lambd, nullptr, "cos_lambda");
        slambd = builder.CreateUnaryIntrinsic(Intrinsic::sin, lambd, nullptr, "sin_lambda");
    } else {
        clambd = ConstantFP::get(realTy, cos(_lambd.value()));
        slambd = ConstantFP::get(realTy, sin(_lambd.value()));
    }

    if (!phi && !lambd) {
        // both are compile-time known
        cphi_lambd = ConstantFP::get(realTy, cos(_phi.value() + _lambd.value()));
        sphi_lambd = ConstantFP::get(realTy, sin(_phi.value() + _lambd.value()));
    } else {
        Value* phiV = (phi) ? phi : ConstantFP::get(realTy, _phi.value());
        Value* lambdV = (lambd) ? lambd : ConstantFP::get(realTy, _lambd.value());
        auto* phi_add_lambd = builder.CreateFAdd(phiV, lambdV, "phi_add_lambda");
        cphi_lambd = builder.CreateUnaryIntrinsic(Intrinsic::cos, phi_add_lambd, nullptr, "cos_phi_add_lambda");
        sphi_lambd = builder.CreateUnaryIntrinsic(Intrinsic::sin, phi_add_lambd, nullptr, "sin_phi_add_lambda");
    }

    // load matrix to vector reg
    // Special optimization only applies when +1, -1, or 0
    auto getFlag = [](std::optional<double> v) -> int {
        if (!v.has_value()) return 2; 
        if (v.value() == 1) return 1;
        if (v.value() == -1) return -1;
        if (v.value() == 0) return 0;
        return 2;
    };

    double _ar = mat.ar.value_or(2);
    double _br = mat.br.value_or(2);
    double _cr = mat.cr.value_or(2);
    double _dr = mat.dr.value_or(2);
    double _bi = mat.bi.value_or(2);
    double _ci = mat.ci.value_or(2);
    double _di = mat.di.value_or(2);

    int arFlag = getFlag(mat.ar);
    int brFlag = getFlag(mat.br);
    int crFlag = getFlag(mat.cr);
    int drFlag = getFlag(mat.dr);
    int biFlag = getFlag(mat.bi);
    int ciFlag = getFlag(mat.ci);
    int diFlag = getFlag(mat.di);

    Value *arElem, *brElem, *crElem, *drElem, *biElem, *ciElem, *diElem;

    errs() << _ar << " " << _br << " " << _cr << " " << _dr << " "
           << _bi << " " << _ci << " " << _di << "\n";

    // ar: cos(theta/2)
    if (mat.ar.has_value()) 
        arElem = ConstantFP::get(realTy, mat.ar.value());
    else 
        arElem = ctheta;
        
    // br: -cos(lambd) sin(theta/2)    
    if (mat.br.has_value()) 
        brElem = ConstantFP::get(realTy, mat.br.value());
    else {
        brElem = builder.CreateFMul(clambd, stheta, "clambd_mul_stheta");
        brElem = builder.CreateFNeg(brElem, "br_elem");
    }

    // cr: cos(phi) sin(theta/2)
    if (mat.cr.has_value()) 
        crElem = ConstantFP::get(realTy, mat.cr.value());
    else 
        crElem = builder.CreateFMul(cphi, stheta, "cr_elem");
    
    // dr: cos(phi+lambda) cos(theta/2)
    if (mat.dr.has_value()) 
        drElem = ConstantFP::get(realTy, mat.dr.value());
    else 
        drElem = builder.CreateFMul(cphi_lambd, ctheta, "dr_elem");

    // bi: -sin(lambd) sin(theta/2)    
    if (mat.bi.has_value()) 
        biElem = ConstantFP::get(realTy, mat.bi.value());
    else {
        biElem = builder.CreateFMul(slambd, stheta, "slambd_mul_stheta");
        biElem = builder.CreateFNeg(biElem, "bi_elem");
    }

    // ci: sin(phi) sin(theta/2)
    if (mat.ci.has_value()) 
        ciElem = ConstantFP::get(realTy, mat.ci.value());
    else 
        ciElem = builder.CreateFMul(sphi, stheta, "ci_elem");
    
    // di: sin(phi+lambda) cos(theta/2)
    if (mat.di.has_value()) 
        diElem = ConstantFP::get(realTy, mat.di.value());
    else 
        diElem = builder.CreateFMul(sphi_lambd, ctheta, "di_elem");

    Value* ar = getVectorWithSameElem(realTy, _S, arElem, "ar");
    Value* br = getVectorWithSameElem(realTy, _S, brElem, "br");
    Value* cr = getVectorWithSameElem(realTy, _S, crElem, "cr");
    Value* dr = getVectorWithSameElem(realTy, _S, drElem, "dr");
    
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

    // idxA = ((idx & outer) << k_sub_s) + ((idx & inner) << s)
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
    newAr = genMulAddOrMulSub(newAr, false, bi, Bi, biFlag, "biBi", "newAr");

    // newAi = ar Ai + ai Ar + br Bi + bi Br
    Value* newAi = nullptr;
    newAi = genMulAddOrMulSub(newAi, true, ar, Ai, arFlag, "arAi", "newAi");
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
