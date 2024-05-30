#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace simulation;

void IRGenerator::loadFromFile(const std::string& fileName) {
    SMDiagnostic err;
    this->mod = parseIRFile(fileName, err, this->llvmContext);
    if (mod == nullptr) {
        err.print("IRGenerator::loadFromFile", llvm::errs());
        this->mod = std::make_unique<Module>("MyModule", this->llvmContext);
    }
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

    // bb is non-special
    if (aa == nullptr)
        return builder.CreateFMul(bb, cc, aaName);
    
    // new_aa = aa + bb * cc
    if (useFMA)
        return builder.CreateIntrinsic(bb->getType(), Intrinsic::fmuladd,
                                       {bb, cc, aa}, nullptr, aaName);
    // not use FMA
    auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
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
