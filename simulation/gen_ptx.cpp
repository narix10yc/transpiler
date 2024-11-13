#include "simulation/ir_generator.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

#include <cmath>
#include <bitset>
#include <algorithm>

using namespace IOColor;
using namespace llvm;
using namespace simulation;
using namespace saot;

Function* IRGenerator::generateCUDAKernel(
        const QuantumGate& gate, const std::string& funcName) {
    
    const auto& qubits = gate.qubits;
    const int nqubits = qubits.size();

    Type* scalarTy = (_config.precision == 32) ? builder.getFloatTy()
                                               : builder.getDoubleTy();
    Function* func;

    Argument *pSvArg, *pMatArg;
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
    SmallVector<Type*> argType { builder.getPtrTy(1U), builder.getPtrTy(4U), };

    auto* funcType = FunctionType::get(builder.getVoidTy(), argType, false);
    func = Function::Create(funcType, Function::ExternalLinkage, funcName, getModule());
    if (funcName == "") {
        std::stringstream ss;
        ss << "ptx_kernel_";
        func->setName(ss.str());
    } else
        func->setName(funcName);
    }

    pSvArg  = func->getArg(0); pSvArg->setName("p.sv");
    pMatArg = func->getArg(1); pMatArg->setName("p.mat");

    // mark this function as a kernel
    auto* mdString = MDString::get(*getContext(), "kernel");
    auto* mdOne = ConstantAsMetadata::get(builder.getInt32(1));
    auto* kernelMetadata = MDNode::get(*getContext(), { ValueAsMetadata::get(func), mdString, mdOne });
    getModule()->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(kernelMetadata);
    
    Value* counterV;
    Value* idxStartV = builder.getInt64(0ULL);

    BasicBlock* entryBB = BasicBlock::Create(*_context, "entry", func);
    builder.SetInsertPoint(entryBB);
    counterV = builder.CreateIntrinsic(builder.getInt32Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, nullptr, "tid.x.i32");
    counterV = builder.CreateIntCast(counterV, builder.getInt64Ty(), true, "tid.x.i64");
    /*
    Example: with target qubits 2, 4, 5
    counter:   xxxhgfedcba
    pbex mask: 11111001011
    idxStart:  hgfed00c0ba

    hgfed00c0ba = (xxxhgfedcba & 00000000011) << 0
                + (xxxhgfedcba & 00000000100) << 1
                + (xxxhgfedcba & 11111111000) << 3
    */

    utils::printVector(gate.qubits, std::cerr << "target qubits: ") << "\n";
    
    // idx = insert 0 to every bit in higherQubits to counter
    idxStartV = builder.getInt64(0ULL);
    uint64_t mask = 0ULL;
    uint64_t maskSum = 0ULL;
    Value* tmpCounterV = counterV;
    for (unsigned i = 0; i < nqubits; i++) {
        unsigned bit = qubits[i];
        mask = ((1ULL << (bit - i)) - 1) - maskSum;
        maskSum = (1ULL << (bit - i)) - 1;
        std::cerr << "i = " << i << ", bit = " << bit
                  << ", mask = " << utils::as0b(mask, 12) << "\n";

        tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
        tmpCounterV = builder.CreateShl(tmpCounterV, i, "tmpCounter");
        idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
    }
    mask = ~((1ULL << (qubits.back() - nqubits + 1)) - 1);
    std::cerr << "mask = " << utils::as0b(mask, 12) << "\n";
    tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = builder.CreateShl(tmpCounterV, nqubits, "tmpCounter");
    idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");

    uint64_t N = 1 << nqubits;
    // std::vector<Value*> reMats(N), imMats(N), reAmpPtrs(N), imAmpPtrs(N), reAmps(N), imAmps(N);
    std::vector<Value*> reAmpPtrs(N), imAmpPtrs(N), reAmps(N), imAmps(N);

    for (uint64_t i = 0; i < N; i++) {
        uint64_t delta = 0ULL;
        for (int b = 0; b < nqubits; b++) {
            if (i & (1 << b))
                delta |= (1 << qubits[b]);
        }
        auto* idxReV = builder.CreateAdd(idxStartV, builder.getInt64(2 * delta), "idx.amp.re." + std::to_string(i));
        reAmpPtrs[i] = builder.CreateGEP(scalarTy, pSvArg, idxReV, "amp.re.ptr." + std::to_string(i));
        reAmps[i] = builder.CreateLoad(scalarTy, reAmpPtrs[i], "amp.re." + std::to_string(i));

        auto* idxImV = builder.CreateAdd(idxStartV, builder.getInt64(2 * delta + 1), "idx.amp.im." + std::to_string(i));
        imAmpPtrs[i] = builder.CreateGEP(scalarTy, pSvArg, idxImV, "amp.im.ptr." + std::to_string(i));
        imAmps[i] = builder.CreateLoad(scalarTy, imAmpPtrs[i], "amp.im." + std::to_string(i));
    }

    for (int r = 0; r < N; r++) {
        // for (int c = 0; c < N; c++) {
        //     // load the r-th row of the matrix
        //     std::string suffix = std::to_string(r) + "." + std::to_string(c);
        //     auto* idxReV = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL * (N*r + c), "idx.mat.re." + suffix);
        //     reMats[c] = builder.CreateLoad(scalarTy, idxReV, "mat.re." + suffix);
        //     auto* idxImV = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL * (N*r + c) + 1, "idx.mat.im." + suffix);
        //     imMats[c] = builder.CreateLoad(scalarTy, idxImV, "mat.im." + suffix);
        // }

        // matrix-vector multiplication
        // Value* updatedReAmp = reAmps[r];
        // Value* updatedImAmp = imAmps[r];
        // for (int c = 0; c < N; c++) {
        //     auto pair = genComplexMultiply({updatedReAmp, updatedImAmp}, {reMats[c], imMats[c]});
        //     updatedReAmp = pair.first;
        //     updatedImAmp = pair.second;
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
        Value* reMatPtr = builder.CreateConstGEP1_64(scalarTy, pMatArg, 0ULL, "mat.re.ptr." + std::to_string(r) + ".0");
        Value* imMatPtr = builder.CreateConstGEP1_64(scalarTy, pMatArg, 1ULL, "mat.im.ptr." + std::to_string(r) + ".0");
        Value* reMat = builder.CreateLoad(scalarTy, reMatPtr, "mat.re." + std::to_string(r) + ".0");
        Value* imMat = builder.CreateLoad(scalarTy, imMatPtr, "mat.im." + std::to_string(r) + ".0");
        Value* updatedReAmp0 = builder.CreateFMul(reAmps[0], reMat);
        Value* updatedReAmp1 = builder.CreateFMul(imAmps[0], imMat);
        Value* updatedImAmp = builder.CreateFMul(reAmps[0], imMat);
        for (int c = 1; c < N; c++) {
            std::string suffix = std::to_string(r) + "." + std::to_string(c);
            reMatPtr = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL * (N*r + c), "idx.mat.re." + suffix);
            reMat = builder.CreateLoad(scalarTy, reMatPtr, "mat.re." + suffix);
            imMatPtr = builder.CreateConstGEP1_64(scalarTy, pMatArg, 2ULL * (N*r + c) + 1, "idx.mat.im." + suffix);
            reMat = builder.CreateLoad(scalarTy, imMatPtr, "mat.im." + suffix);

            updatedReAmp0 = builder.CreateIntrinsic(
                scalarTy, Intrinsic::fmuladd, { reAmps[c], reMat, updatedReAmp0 });
            updatedReAmp1 = builder.CreateIntrinsic(
                scalarTy, Intrinsic::fmuladd, { imAmps[c], imMat, updatedReAmp1 });
            updatedImAmp = builder.CreateIntrinsic(
                scalarTy, Intrinsic::fmuladd, { reAmps[c], imMat, updatedImAmp });
            updatedImAmp = builder.CreateIntrinsic(
                scalarTy, Intrinsic::fmuladd, { imAmps[c], reMat, updatedImAmp });
        }
        Value* updatedReAmp = builder.CreateFSub(updatedReAmp0, updatedReAmp1);
        builder.CreateStore(updatedReAmp, reAmpPtrs[r]);
        builder.CreateStore(updatedImAmp, imAmpPtrs[r]);
    }

    builder.CreateRetVoid();
    return func;
}
