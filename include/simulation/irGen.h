#ifndef SIMULATION_CODEGEN_H_
#define SIMULATION_CODEGEN_H_

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include <cmath>
#include <memory>
#include <vector>
#include <array>
#include <optional>

namespace simulation::codegen {

// static std::optional<double> optionalCos(std::optional<double> a) {
//     if (!a.has_value()) return {};
//     return { cos(a.value()) };
// }

// static std::optional<double> optionalSin(std::optional<double> a) {
//     if (!a.has_value()) return {};
//     return { sin(a.value()) };
// }

static double approxCos(double a, double thres=1e-8) {
    double v = cos(a);
    if (abs(v) < thres) { return 0; }
    if (abs(v - 1) < thres) { return 1; }
    if (abs(v + 1) < thres) { return -1; }
    return v; 
}

static double approxSin(double a, double thres=1e-8) {
    double v = sin(a);
    if (abs(v) < thres) { return 0; }
    if (abs(v - 1) < thres) { return 1; }
    if (abs(v + 1) < thres) { return -1; }
    return v; 
}

class OptionalComplexMat2x2 {
public:
    std::optional<double> ar, br, cr, dr, bi, ci, di;
public:
    OptionalComplexMat2x2(std::optional<double> ar, 
    std::optional<double> br, std::optional<double> cr,
    std::optional<double> dr, std::optional<double> bi,
    std::optional<double> ci, std::optional<double> di) 
    : ar(ar), br(br), cr(cr), dr(dr), bi(bi), ci(ci), di(di) {}

    static OptionalComplexMat2x2 
    FromAngles(std::optional<double> theta, std::optional<double> phi, 
               std::optional<double> lambd, double thres=1e-8) {
        std::optional<double> ar, br, cr, dr, bi, ci, di;
        std::optional<double> ctheta, stheta, clambd, slambd, cphi, sphi,
                              cphi_lambd, sphi_lambd;
        if (theta.has_value()) {
            ctheta = approxCos(theta.value() * 0.5, thres);
            stheta = approxSin(theta.value() * 0.5, thres);
        }
        if (lambd.has_value()) {
            clambd = approxCos(lambd.value(), thres);
            slambd = approxSin(lambd.value(), thres);
        }
        if (phi.has_value()) {
            cphi = approxCos(phi.value(), thres);
            sphi = approxSin(phi.value(), thres);
        }
        if (phi.has_value() && lambd.has_value()) {
            cphi_lambd = approxCos(phi.value() + lambd.value(), thres);
            sphi_lambd = approxSin(phi.value() + lambd.value(), thres);
        }

        // ar: cos(theta/2)
        if (ctheta.has_value())
            ar.emplace(ctheta.value());

        // br: -cos(lambd) * sin(theta/2)
        if (clambd.has_value() && stheta.has_value()) {
            br.emplace(-clambd.value() * stheta.value());
        }
        if ((clambd.has_value() && clambd == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            br.emplace(0);
        }

        // cr: cos(phi) * sin(theta/2)
        if (cphi.has_value() && stheta.has_value()) {
            cr.emplace(cphi.value() * stheta.value());
        }
        if ((cphi.has_value() && cphi == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            cr.emplace(0);
        }

        // dr: cos(phi+lambd) * cos(theta/2)
        if (cphi_lambd.has_value() && ctheta.has_value()) {
            dr.emplace(cphi_lambd.value() * ctheta.value());
        }
        if ((cphi_lambd.has_value() && cphi_lambd == 0) || 
                 (ctheta.has_value() && ctheta == 0)) {
            dr.emplace(0);
        }

        // bi: -sin(lambd) * sin(theta/2)
        if (slambd.has_value() && stheta.has_value()) {
            bi.emplace(-slambd.value() * stheta.value());
        }
        if ((slambd.has_value() && slambd == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            bi.emplace(0);
        }

        // ci: sin(phi) * sin(theta/2)
        if (sphi.has_value() && stheta.has_value()) {
            ci.emplace(sphi.value() * stheta.value());
        }
        if ((sphi.has_value() && sphi == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            ci.emplace(0);
        }

        // di: sin(phi+lambd) * cos(theta/2)
        if (sphi_lambd.has_value() && ctheta.has_value()) {
            di.emplace(sphi_lambd.value() * ctheta.value());
        }
        if ((sphi_lambd.has_value() && sphi_lambd == 0) || 
                 (ctheta.has_value() && ctheta == 0)) {
            di.emplace(0);
        }
        return { ar, br, cr, dr, bi, ci, di };
    }
};
    
class IRGenerator {
    llvm::LLVMContext llvmContext;
    llvm::IRBuilder<> builder;
    std::unique_ptr<llvm::Module> mod;

    // parameters
    int vector_size_in_bits;
private:
    llvm::Value* getVectorWithSameElem(llvm::Type* realTy, const unsigned length, 
            llvm::Value* elem, const llvm::Twine &name = "") {
        llvm::Value* vec = llvm::UndefValue::get(llvm::VectorType::get(realTy, length, false));
        for (size_t i = 0; i < length - 1; ++i) {
            vec = builder.CreateInsertElement(vec, elem, i, name + "_insert_" + std::to_string(i));
        }
        vec = builder.CreateInsertElement(vec, elem, length - 1, name + "_vec");  
        return vec;  
    }

    llvm::Value* getVectorWithSameElem(llvm::Type* realTy, const unsigned length, 
            const double value, const llvm::Twine &name = "") {
        auto* elem = llvm::ConstantFP::get(llvmContext, llvm::APFloat(value));
        return getVectorWithSameElem(realTy, length, elem, name);
    }

public:
    IRGenerator() : 
        builder(llvmContext), 
        mod(std::make_unique<llvm::Module>("myModule", llvmContext)) {
        vector_size_in_bits = 2;
    }

    void setVectorSizeInBits(const unsigned bits) {
        vector_size_in_bits = bits;
    }

    enum RealTy : int { Float, Double };
    
    llvm::Module& getModule() const { return *mod; }

    /// @brief Generate the IR that applies new_aa = aa +/- bb * cc
    /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
    /// @param bb
    /// @param cc 
    /// @param bbFlag either Add (for bb = 1), Sub (for bb = -1), Zero (for bb 
    ///   = 0), or Normal (otherwise)
    /// @return aa + bb * cc
    llvm::Value* genMulAddOrMulSub(
            llvm::Value* aa,
            bool add,
            llvm::Value* bb,
            llvm::Value* cc,
            int bbFlag,
            const llvm::Twine& bbccName = "",
            const llvm::Twine& aaName = "");

    void genU3(const int64_t k, 
            const llvm::StringRef funcName, 
            const RealTy realType,
            std::optional<double> theta, 
            std::optional<double> phi, 
            std::optional<double> lambd, 
            double thres=1e-8);
};


} // namespace simulation::codegen

#endif // SIMULATION_CODEGEN_H_