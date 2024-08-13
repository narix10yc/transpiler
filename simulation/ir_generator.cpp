#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "utils/iocolor.h"

using namespace Color;
using namespace llvm;
using namespace simulation;

bool IRGeneratorConfig::checkConfliction(std::ostream& os) const {
    bool check = true;
    const auto warn = [&os]() -> std::ostream& {
        return os << YELLOW_FG << BOLD << "Warning: " << RESET;
    };
    if (shareMatrixElemThres < 0.0) {
        warn() << "Set 'shareMatrixElemThres' to a negative value has no effect\n";
        check = false;
    }
    return check;
}

std::ostream& IRGeneratorConfig::display(
        int verbose, bool title, std::ostream& os) const {
    if (title)
        os << CYAN_FG << "=== IR Generator Config ===\n" << RESET;
    
    const char* ON = "\033[32mon\033[0m";
    const char* OFF = "\033[31moff\033[0m";

    os << "simd s:               " << simd_s << "\n"
       << "precision:            " << "f" << precision << "\n"
       << "amp format:           " << ((ampFormat == AmpFormat::Alt) ? "Alt" : "Sep") << "\n"
       << "FMA " << ((useFMA) ? ON : OFF)
       << ", FMS " << ((useFMS) ? ON : OFF)
       << ", PDEP " << ((usePDEP) ? ON : OFF) << "\n"
       << "loadMatrixInEntry:    " << ((loadMatrixInEntry) ? "true" : "false") << "\n"
       << "loadVectorMatrix:     " << ((loadVectorMatrix) ? "true" : "false") << "\n"
       << "forceDenseKernel:     " << ((forceDenseKernel) ? "true" : "false") << '\n'
       << "zeroSkipThres:        " << std::scientific << zeroSkipThres << "\n"
       << "shareMatrixElemThres: " << std::scientific << shareMatrixElemThres << "\n"
       << "shareMatrixElemUseImmValue " << ((shareMatrixElemUseImmValue) ? ON : OFF) << "\n";

    if (title)
       os << CYAN_FG << "===========================\n" << RESET;
    return os;
}

void IRGenerator::loadFromFile(const std::string& fileName) {
    SMDiagnostic err;
    this->mod = parseIRFile(fileName, err, this->llvmContext);
    if (mod == nullptr) {
        err.print("IRGenerator::loadFromFile", llvm::errs());
        this->mod = std::make_unique<Module>("MyModule", this->llvmContext);
    }
}

Value* IRGenerator::genMulAdd(
        Value* aa, Value* bb, Value* cc, int bbFlag,
        const Twine& bbccName, const Twine& aaName) {
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
    if (_config.useFMA)
        return builder.CreateIntrinsic(bb->getType(), Intrinsic::fmuladd,
                                       {bb, cc, aa}, nullptr, aaName);
    // not use FMA
    auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
    return builder.CreateFAdd(aa, bbcc, aaName);
}


Value* IRGenerator::genMulSub(
        Value* aa, Value* bb, Value* cc, int bbFlag,
        const Twine& bbccName, const Twine& aaName) {
    if (bbFlag == 0) 
        return aa;

    auto* ccNeg = builder.CreateFNeg(cc, "ccNeg");
    // new_aa = aa - cc
    if (bbFlag == 1) {
        if (aa == nullptr)
            return ccNeg;
        return builder.CreateFSub(aa, cc, aaName);
    }

    // new_aa = aa + cc
    if (bbFlag == -1) {
        if (aa == nullptr)
            return cc;
        return builder.CreateFAdd(aa, cc, aaName);
    }

    // bb is non-special
    // new_aa = aa - bb * cc
    if (aa == nullptr)
        return builder.CreateFMul(bb, ccNeg, aaName);

    if (_config.useFMS)
        return builder.CreateIntrinsic(bb->getType(), Intrinsic::fmuladd,
                                       {bb, ccNeg, aa}, nullptr, aaName);
    // not use FMS
    auto* bbccNeg = builder.CreateFMul(bb, ccNeg, bbccName + "Neg");
    return builder.CreateFAdd(aa, bbccNeg, aaName);
}
