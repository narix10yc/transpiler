#include "simulation/ir_generator.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

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

    /// @return (mask, vec)
    std::pair<std::vector<int>, std::vector<int>>
    getMaskToMerge(const std::vector<int>& v0, const std::vector<int>& v1) {
        assert(v0.size() == v1.size());
        const auto s = v0.size();
        std::vector<int> mask(2*s);
        std::vector<int> vec(2*s);
        unsigned i0 = 0, i1 = 0, i;
        int elem0, elem1;
        while (i0 < s || i1 < s) {
            i = i0 + i1;
            if (i0 == s) {
                vec[i] = v1[i1];
                mask[i] = i1 + s;
                i1++;
                continue;
            }
            if (i1 == s) {
                vec[i] = v0[i0];
                mask[i] = i0;
                i0++;
                continue;
            }
            elem0 = v0[i0];
            elem1 = v1[i1];
            if (elem0 < elem1) {
                vec[i] = elem0;
                mask[i] = i0;
                i0++;
            } else {
                vec[i] = elem1;
                mask[i] = i1 + s;
                i1++;
            }
        }
        return std::make_pair(mask, vec);
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

    SmallVector<StringRef> argNames
        { "preal", "pimag", "counter_start", "counter_end", "pmat"};
    unsigned i = 0;
    for (auto& arg : func->args())
        arg.setName(argNames[i++]);

    auto* pRealArg = func->getArg(0);
    auto* pImagArg = func->getArg(1);
    auto* counterStartArg = func->getArg(2);
    auto* counterEndArg = func->getArg(3);
    auto* pMatArg = func->getArg(4);

    // init basic blocks
    BasicBlock* entryBB = BasicBlock::Create(llvmContext, "entry", func);
    BasicBlock* loopBB = BasicBlock::Create(llvmContext, "loop", func);
    BasicBlock* loopBodyBB = BasicBlock::Create(llvmContext, "loopBody", func);
    BasicBlock* retBB = BasicBlock::Create(llvmContext, "ret", func);

    builder.SetInsertPoint(entryBB);

    // load matrix
    std::vector<matrix_data_t> matrix(K * K);
    auto* matV = builder.CreateLoad(VectorType::get(scalarTy, 2 * K * K, false),
                                    pMatArg, "matrix");
    for (unsigned i = 0; i < matrix.size(); i++) {
        matrix[i].realVal = builder.CreateShuffleVector(
            matV, std::vector<int>(S, 2*i), "mRe_" + std::to_string(i));
        matrix[i].imagVal = builder.CreateShuffleVector(
            matV, std::vector<int>(S, 2*i+1), "mIm_" + std::to_string(i));

        auto real = gate.matrix.matrix.constantMatrix.data.at(i).real();
        auto imag = gate.matrix.matrix.constantMatrix.data.at(i).imag();

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
    
    // debug print matrix
    if (verbose > 1) {
        std::cerr << "IRMatrix:\n[";
        for (unsigned r = 0; r < K; r++) {
            for (unsigned c = 0; c < K; c++) {
                int realFlag = matrix[r*K + c].realFlag;
                int imagFlag = matrix[r*K + c].imagFlag;
                std::cerr << "(" << realFlag << "," << imagFlag << "),";
            }
            if (r < K - 1)
                std::cerr << "\n ";
            else
                std::cerr << "]\n";
        }
    }

    // split qubits
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
    sepBit++; // separation bit = lk + s
    unsigned vecSize = 1U << sepBit;
    auto* vecType = VectorType::get(scalarTy, vecSize, false);

    const unsigned lk = lowerQubits.size();
    const unsigned LK = 1 << lk;
    const unsigned hk = higherQubits.size();
    const unsigned HK = 1 << hk;

    // debug print qubit splits
    if (verbose > 1) {
        std::cerr << "simdQubits: ";
        printVector(simdQubits) << "\n";
        std::cerr << "lowerQubits: ";
        printVector(lowerQubits) << "\n";
        std::cerr << "higherQubits: ";
        printVector(higherQubits) << "\n";
        std::cerr << "sepBit: " << sepBit << "\n";
        std::cerr << "vecSize: " << vecSize << "\n";
    }

    builder.CreateBr(loopBB);

    // loop entry
    builder.SetInsertPoint(loopBB);
    PHINode* counterV = builder.CreatePHI(builder.getInt64Ty(), 2, "counter");
    counterV->addIncoming(counterStartArg, entryBB);
    Value* cond = builder.CreateICmpSLT(counterV, counterEndArg, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);

    // find start pointer
    if (verbose > 2) {
        std::cerr << "finding masks... sum of ((counter & mask) << i)\n";
    }
    Value* idxStartV = builder.getInt64(0ULL);
    if (!higherQubits.empty()) {
        // idx = insert 0 to every bit in higherQubits to counter
        uint64_t mask = 0ULL;
        Value* tmpCounterV = counterV;
        for (unsigned i = 0; i < higherQubits.size(); i++) {
            unsigned bit = higherQubits[i];
            mask = ((1ULL << (bit - sepBit - i)) - 1) - mask;
            if (verbose > 2) {
                std::cerr << "i = " << i << ", bit = " << bit
                        << ", mask = " << std::bitset<12>(mask) << "\n";
            }
            tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
            tmpCounterV = builder.CreateShl(tmpCounterV, i, "tmpCounter");
            idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
        }
        mask = ~((1ULL << (higherQubits.back() - sepBit - higherQubits.size() + 1)) - 1);
        if (verbose > 2) {
            std::cerr << "                mask = "
                    << std::bitset<12>(mask) << "\n";
        }
        tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
        tmpCounterV = builder.CreateShl(tmpCounterV, higherQubits.size(), "tmpCounter");
        idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");
    }

    // split masks, to be used in loading amplitude registers
    std::vector<std::vector<int>> splits(LK);
    for (size_t i = 0; i < vecSize; i++) {
        unsigned key = 0;
        for (unsigned lowerI = 0; lowerI < lk; lowerI++) {
            if (i & (1 << lowerQubits[lowerI]))
                key |= (1 << lowerI);
        }
        splits[key].push_back(i);
    }
    // debug print splits
    if (verbose > 1) {
        std::cerr << "splits: [";
        for (const auto& split : splits) {
            std::cerr << "\n ";
            printVector(split);
        }
        std::cerr << "\n]\n";
    }
    
    // load amplitude registers
    std::vector<Value*> real(K, nullptr), imag(K, nullptr);
    std::vector<Value*> pReal(HK, nullptr), pImag(HK, nullptr);
    Value *realFull, *imagFull;
    for (unsigned hi = 0; hi < HK; hi++) {
        unsigned keyStart = 0;
        uint64_t idxShift = 0ULL;
        for (unsigned higherI = 0; higherI < hk; higherI++) {
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
        auto* _idxV = builder.CreateAdd(idxStartV, builder.getInt64(idxShift), "idx_" + std::to_string(hi));
        pReal[hi] = builder.CreateGEP(vecType, pRealArg, _idxV, "pReal_" + std::to_string(hi));
        pImag[hi] = builder.CreateGEP(vecType, pImagArg, _idxV, "pImag_" + std::to_string(hi));
        realFull = builder.CreateLoad(vecType, pReal[hi], "reFull_" + std::to_string(hi));
        imagFull = builder.CreateLoad(vecType, pImag[hi], "imFull_" + std::to_string(hi));
        for (unsigned i = 0; i < LK; i++) {
            unsigned key = keyStart + i;
            real[key] = builder.CreateShuffleVector(
                realFull, splits[i], "real_" + std::to_string(key));
            imag[key] = builder.CreateShuffleVector(
                imagFull, splits[i], "imag_" + std::to_string(key));
        }
    }
    
    // matrix-vector multiplication
    std::vector<Value*> newReal(K, nullptr);
    for (unsigned r = 0; r < K; r++) {
        std::string name0 = "newRe0_" + std::to_string(r) + "_";
        std::string name1 = "newRe1_" + std::to_string(r) + "_";

        Value *newRe0 = nullptr, *newRe1 = nullptr;
        for (unsigned c = 0; c < K; c++) {
            newRe0 = genMulAdd(newRe0, matrix[r * K + c].realVal, real[c],
                               matrix[r * K + c].realFlag, "", name0);
            newRe1 = genMulAdd(newRe1, matrix[r * K + c].imagVal, imag[c],
                               matrix[r * K + c].imagFlag, "", name1);  
        }
        if (newRe0 != nullptr && newRe1 != nullptr)
            newReal[r] = builder.CreateFSub(newRe0, newRe1, "newRe" + std::to_string(r) + "_");
        else if (newRe0 == nullptr)
            newReal[r] = builder.CreateFNeg(newRe1, "newRe" + std::to_string(r) + "_");
        else if (newRe1 == nullptr)
            newReal[r] = newRe0;
        else
            assert(false && "newReal is null");
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

    // merge updated amplitudes
    std::vector<std::vector<int>> mergeMasks;
    for (unsigned round = 0; round < lk; round++) {
        for (unsigned pairI = 0; pairI < (1 << (lk - round - 1)); pairI++) {
            auto pair = getMaskToMerge(splits[2*pairI], splits[2*pairI + 1]);
            mergeMasks.push_back(std::move(pair.first));
            splits[pairI] = std::move(pair.second);
        }
    }
    std::cerr << "mergeMasks:\n";
    for (const auto& vec : mergeMasks)
        printVector(vec) << "\n";
    std::cerr << "\n";

    // merge and store back
    for (unsigned hi = 0; hi < HK; hi++) {
        unsigned mergeBegin = hi << lk;
        auto maskIt = mergeMasks.cbegin();
        std::cerr << "hi = " << hi << ", "
                  << "mergeBegin = " << mergeBegin << "\n";
        for (unsigned round = 0; round < lk; round++) {
            std::cerr << " round " << round
                      << ", there are " << (1 << (lk - round - 1)) << " pairs\n";
            for (unsigned pairI = 0; pairI < (1 << (lk - round - 1)); pairI++) {
                unsigned mergeIdx = mergeBegin + 2*pairI;
                unsigned storeIdx = mergeBegin + pairI;
                std::cerr << "  pairI = " << pairI << ", "
                          << "pair = (" << mergeIdx << "," << mergeIdx + 1 << ")"
                          << " => " << storeIdx << "\n";
                newReal[storeIdx] = builder.CreateShuffleVector(
                        newReal[mergeIdx], newReal[mergeIdx + 1],
                        *maskIt, "mergeRe");
                newImag[storeIdx] = builder.CreateShuffleVector(
                        newImag[mergeIdx], newImag[mergeIdx + 1],
                        *maskIt, "mergeIm");
                maskIt++;
            }
        }
        std::cerr << "store " << mergeBegin << " to pReal " << hi << "\n";
        builder.CreateStore(newReal[mergeBegin], pReal[hi], false);
        builder.CreateStore(newImag[mergeBegin], pImag[hi], false);
    }

    // increment counter and return 
    auto* counterNextV = builder.CreateAdd(counterV, builder.getInt64(1), "counterNext");
    counterV->addIncoming(counterNextV, loopBodyBB);
    builder.CreateBr(loopBB);
    
    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}