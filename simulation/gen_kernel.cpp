#include "simulation/ir_generator.h"
#include "utils/iocolor.h"

using namespace llvm;
using namespace Color;
using namespace simulation;
using namespace quench::quantum_gate;

namespace {
    struct matrix_data_t {
        Value* realVal;
        Value* imagVal;
        int realFlag;
        int imagFlag;
    };

    template<typename T>
    std::ostream& printVector(const std::vector<T>& v, std::ostream& os = std::cerr) {
        if (v.empty())
            return os << "[]";
        os << "[";
        for (unsigned i = 0; i < v.size() - 1; i++)
            os << v[i] << ",";
        os << v.back() << "]";
        return os;
    }
}

Function*
IRGenerator::generateKernel(const QuantumGate& gate,
                            const std::string& funcName)
{
    const uint64_t s = vecSizeInBits;
    const uint64_t S = 1ULL << s;
    const uint64_t k = gate.qubits.size();
    const uint64_t K = 1ULL << k;

    if (verbose > 0) {
        std::cerr << CYAN_FG << "== Generating Kernel '"
                  << funcName << "' ==" <<  RESET << "\n"
                  << "s = " << s << "; S = " << S << "\n"
                  << "k = " << k << "; K = " << K << "\n"
                  << "qubits = ";
        printVector(gate.qubits) << "\n";
    }

    Type* scalarTy = (realTy == ir::RealTy::Float) ? builder.getFloatTy()
                                                   : builder.getDoubleTy();
    Type* retTy = builder.getVoidTy();
    SmallVector<Type*> argTy;
    argTy.push_back(builder.getPtrTy()); // ptr to real amp
    argTy.push_back(builder.getPtrTy()); // ptr to imag amp
    argTy.push_back(builder.getInt64Ty()); // counter_start
    argTy.push_back(builder.getInt64Ty()); // counter_end
    argTy.push_back(builder.getPtrTy()); // ptr to matrix

    FunctionType* funcTy = FunctionType::get(retTy, argTy, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage, funcName, mod.get());

    auto* pRealArg = func->getArg(0);
    auto* pImagArg = func->getArg(1);
    auto* counterStartArg = func->getArg(2);
    auto* counterEndArg = func->getArg(3);
    auto* pMatArg = func->getArg(4);

    SmallVector<StringRef> argNames
        { "preal", "pimag", "counter_start", "counter_end", "pmat"};

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

    // set up matrix
    std::vector<matrix_data_t> matrix(K * K);
    auto* matV = builder.CreateLoad(VectorType::get(scalarTy, 2 * K * K, false),
                                    pMatArg, "matrix");
    for (unsigned i = 0; i < matrix.size(); i++) {
        matrix[i].realVal = builder.CreateShuffleVector(
            matV, std::vector<int>(S, 2*i), "mRe_" + std::to_string(i));
        matrix[i].imagVal = builder.CreateShuffleVector(
            matV, std::vector<int>(S, 2*i+1), "mIm_" + std::to_string(i));

        auto real = gate.matrix.matrix.constantMatrix.data.at(i).real;
        auto imag = gate.matrix.matrix.constantMatrix.data.at(i).imag;

        double thres = 1e-8;
        if (std::abs(real) < thres)
            matrix[i].realFlag = 0;
        else if (std::abs(real - 1.0) < thres)
            matrix[i].realFlag = 1;
        else if (std::abs(real + 1.0) < thres)
            matrix[i].realFlag = -1;
        else 
            matrix[i].realFlag = 2;

        if (std::abs(imag) < thres)
            matrix[i].imagFlag = 0;
        else if (std::abs(imag - 1.0) < thres)
            matrix[i].imagFlag = 1;
        else if (std::abs(imag + 1.0) < thres)
            matrix[i].imagFlag = -1;
        else 
            matrix[i].imagFlag = 2;
    }

    unsigned _q = 0;
    auto qubitsIt = gate.qubits.cbegin();
    std::vector<unsigned> simdQubits, higherQubits, lowerQubits;
    while (simdQubits.size() != s) {
        if (qubitsIt != gate.qubits.cend() && *qubitsIt == _q) {
            lowerQubits.push_back(_q);
            qubitsIt++;
        } else {
            simdQubits.push_back(_q);
        }
        _q++;
    }
    while (qubitsIt != gate.qubits.cend()) {
        higherQubits.push_back(*qubitsIt);
        qubitsIt++;
    }
    unsigned sepBit = simdQubits.back();
    if (!lowerQubits.empty() && lowerQubits.back() > sepBit)
        sepBit = lowerQubits.back();
    sepBit++;

    if (verbose > 1) {
        std::cerr << "simdQubits: ";
        printVector(simdQubits) << "\n";
        std::cerr << "lowerQubits: ";
        printVector(lowerQubits) << "\n";
        std::cerr << "higherQubits: ";
        printVector(higherQubits) << "\n";
        std::cerr << "sepBit: " << sepBit << "\n";
    }

    builder.CreateBr(loopBB);

    // loop
    builder.SetInsertPoint(loopBB);
    PHINode* counterV = builder.CreatePHI(builder.getInt64Ty(), 2, "counter");
    counterV->addIncoming(counterStartArg, entryBB);
    Value* cond = builder.CreateICmpSLT(counterV, counterEndArg, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    // load amplitude registers
    Value* idxV = nullptr;
    std::vector<Value*> real(K, nullptr);
    std::vector<Value*> imag(K, nullptr);
    Value *pReal, *pImag, *realFull, *imagFull;
    unsigned vecSize = 1U << sepBit;
    auto* vecType = VectorType::get(scalarTy, vecSize, false);
    if (higherQubits.empty()) {
        assert(k == lowerQubits.size());

        idxV = counterV;
        pReal = builder.CreateGEP(vecType, pRealArg, idxV, "pReal");
        pImag = builder.CreateGEP(vecType, pImagArg, idxV, "pImag");
        realFull = builder.CreateLoad(vecType, pReal, "reFull");
        imagFull = builder.CreateLoad(vecType, pImag, "imFull");

        std::vector<std::vector<int>> splits(K);
        for (size_t i = 0; i < vecSize; i++) {
            unsigned key = 0;
            for (unsigned lowerI = 0; lowerI < k; lowerI++) {
                if (i & (1 << lowerQubits[lowerI]))
                    key |= (1 << lowerI);
            }
            splits[key].push_back(i);
        }
        if (verbose > 1) {
            std::cerr << "splits: [";
            for (const auto& split : splits) {
                std::cerr << "\n ";
                printVector(split);
            }
            std::cerr << "\n]\n";
        }
        for (unsigned i = 0; i < K; i++) {
            real[i] = builder.CreateShuffleVector(
                realFull, splits[i], "real_" + std::to_string(i));
            imag[i] = builder.CreateShuffleVector(
                imagFull, splits[i], "imag_" + std::to_string(i));
        }
    }
    else {
        // idx = insert 0 to every bit in higherQubits to counter
        uint64_t mask = 0ULL;
        Value* tmpCounterV = counterV;
        Value* idxStartV = builder.getInt64(0ULL);
        for (unsigned i = 0; i < higherQubits.size(); i++) {
            mask = ((1ULL << (higherQubits[i] - sepBit - i)) - 1) - mask;
            if (verbose > 2) {
                std::cerr << "i = " << i << ", bit = " << higherQubits[i]
                          << ", mask = " << std::bitset<12>(mask) << "\n";
            }
            tmpCounterV = builder.CreateAnd(tmpCounterV, mask, "tmpCounter");
            tmpCounterV = builder.CreateShl(tmpCounterV, i, "tmpCounter");
            idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
        }
        mask = ~((1ULL << (higherQubits.back() - sepBit - higherQubits.size() + 1)) - 1);
        if (verbose > 2) {
            std::cerr << "                mask = "
                      << std::bitset<12>(mask) << "\n";
        }
        tmpCounterV = builder.CreateAnd(tmpCounterV, mask, "tmpCounter");
        tmpCounterV = builder.CreateShl(tmpCounterV, higherQubits.size(), "tmpCounter");
        idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");

        std::vector<std::vector<int>> splits(1 << lowerQubits.size());
        for (size_t i = 0; i < vecSize; i++) {
            unsigned key = 0;
            for (unsigned lowerI = 0; lowerI < lowerQubits.size(); lowerI++) {
                if (i & (1 << lowerQubits[lowerI]))
                    key |= (1 << lowerI);
            }
            splits[key].push_back(i);
        }
        if (verbose > 1) {
            std::cerr << "splits: [";
            for (const auto& split : splits) {
                std::cerr << "\n ";
                printVector(split);
            }
            std::cerr << "\n]\n";
        }

        for (unsigned hi = 0; hi < (1 << higherQubits.size()); hi++) {
            unsigned keyStart = 0;
            uint64_t idxShift = 0ULL;
            for (unsigned higherI = 0; higherI < higherQubits.size(); higherI++) {
                if (hi & (1 << higherI)) {
                    idxShift += 1ULL << (higherQubits[higherI] - sepBit);
                    keyStart += 1 << (higherI + lowerQubits.size());
                }
            }
            if (verbose > 2) {
                std::cerr << "hi = " << hi << ", "
                          << "idxShift = " << std::bitset<12>(idxShift) << ", "
                          << "keyStart = " << keyStart << "\n";
            }
            idxV = builder.CreateAdd(idxStartV, builder.getInt64(idxShift), "idx");
            pReal = builder.CreateGEP(vecType, pRealArg, idxV, "pReal");
            pImag = builder.CreateGEP(vecType, pImagArg, idxV, "pImag");
            realFull = builder.CreateLoad(vecType, pReal, "reFull");
            imagFull = builder.CreateLoad(vecType, pImag, "imFull");
            for (unsigned i = 0; i < (1 << lowerQubits.size()); i++) {
                unsigned key = keyStart + i;
                if (verbose > 2) {
                    std::cerr << "key = " << key << "\n";
                    printVector(splits[i]) << "\n";
                }
                real[key] = builder.CreateShuffleVector(
                    realFull, splits[i], "real_" + std::to_string(key));
                imag[key] = builder.CreateShuffleVector(
                    imagFull, splits[i], "imag_" + std::to_string(key));
            }
        }
    }

    // matrix-vector multiplication
    std::vector<Value*> newReal(K, nullptr);
    for (unsigned r = 0; r < K; r++) {
        std::string name0 = "newRe0_" + std::to_string(r) + "_";
        std::string name1 = "newRe1_" + std::to_string(r) + "_";

        Value *newRe0 = nullptr, *newRe1 = nullptr;
        for (unsigned c = 0; c < K; c++) {
            // std::cerr << "r*K+c = " << r*K+c << ", "
            //           << "realFlag = " << matrix[r * K + c].realFlag << ", "
            //           << "imagFlag = " << matrix[r * K + c].imagFlag << "\n";
            newRe0 = genMulAdd(newRe0, matrix[r * K + c].realVal, real[c],
                               matrix[r * K + c].realFlag, "", name0);
            newRe1 = genMulAdd(newRe1, matrix[r * K + c].imagVal, imag[c],
                               matrix[r * K + c].imagFlag, "", name1);  
        }
        if (newRe0 != nullptr && newRe1 != nullptr)
            newReal[r] = builder.CreateFSub(newRe0, newRe1, "newRe" + std::to_string(r));
        else if (newRe0 == nullptr)
            newReal[r] = builder.CreateFNeg(newRe1, "newRe" + std::to_string(r));
        else if (newRe1 == nullptr)
            newReal[r] = newRe0;
        else
            assert(false);
    }

    std::vector<Value*> newImag(K, nullptr);
    for (unsigned r = 0; r < K; r++) {
        std::string name = "newIm" + std::to_string(r) + "_";
        for (unsigned c = 0; c < K; c++) {
            newImag[r] = genMulAdd(newImag[r], matrix[r * K + c].realVal, imag[c],
                               matrix[r * K + c].realFlag, "", name);
            newImag[r] = genMulAdd(newImag[r], matrix[r * K + c].imagVal, real[c],
                               matrix[r * K + c].imagFlag, "", name);
        }
    }
    // store amplitudes



    // increment counter and return 
    auto* counterNextV = builder.CreateAdd(counterV, builder.getInt64(1), "counterNext");
    counterV->addIncoming(counterNextV, loopBodyBB);
    builder.CreateBr(loopBB);
    
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}