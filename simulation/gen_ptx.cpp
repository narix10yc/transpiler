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

    Value* counterV;
    Value* idxStartV = builder.getInt64(0ULL);

    BasicBlock* entryBB = BasicBlock::Create(*_context, "entry", func);
    builder.SetInsertPoint(entryBB);
    counterV = builder.CreateIntrinsic(builder.getInt64Ty(), Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, nullptr, "tid.x");

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
    for (unsigned i = 0; i < gate.qubits.size(); i++) {
        unsigned bit = gate.qubits[i];
        mask = ((1ULL << (bit - i)) - 1) - maskSum;
        maskSum = (1ULL << (bit - i)) - 1;
        std::cerr << "i = " << i << ", bit = " << bit
                  << ", mask = " << utils::as0b(mask, 12) << "\n";

        tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
        tmpCounterV = builder.CreateShl(tmpCounterV, i, "tmpCounter");
        idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
    }
    mask = ~((1ULL << (gate.qubits.back() - gate.qubits.size() + 1)) - 1);
    std::cerr << "mask = " << utils::as0b(mask, 12) << "\n";
    tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
    tmpCounterV = builder.CreateShl(tmpCounterV, gate.qubits.size(), "tmpCounter");
    idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");


    return func;
}
