#include "simulation/ir_generator.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/IR/IntrinsicsX86.h"

#include <cmath>
#include <bitset>
#include <algorithm>

using namespace utils;
using namespace llvm;
using namespace IOColor;
using namespace simulation;
using namespace saot;

namespace { /* anonymous namespace */
    struct matrix_data_t {
        Value* realVal;
        Value* imagVal;
        int realFlag;
        int imagFlag;
        bool realLoadNeg;
        bool imagLoadNeg;
    };

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
} // anonymous namespace

Function*
IRGenerator::generateKernelDebug(
        const QuantumGate& gate, int debugLevel, const std::string& funcName) {

    const uint64_t s = _config.simd_s;
    const uint64_t S = 1ULL << s;
    const uint64_t k = gate.qubits.size();
    const uint64_t K = 1ULL << k;

    if (debugLevel > 0) {
        std::cerr << CYAN_FG << BOLD << "[IR Generation Debug] " << RESET
                  << funcName << "\n"
                  << "s = " << s << "; S = " << S << "\n"
                  << "k = " << k << "; K = " << K << "\n"
                  << "target qubits: ";
        printVector(gate.qubits) << "\n";
    }

    Type* scalarTy = (_config.precision == 32) ? builder.getFloatTy()
                                               : builder.getDoubleTy();
    Function* func;
    Argument *pSvArg, *pReArg, *pImArg, *ctrBeginArg, *ctrEndArg, *pMatArg;
    { /* start of function declaration */
    auto argType = (_config.ampFormat == AmpFormat::Sep)
            ? SmallVector<Type*> { builder.getPtrTy(), builder.getPtrTy(),
                builder.getInt64Ty(), builder.getInt64Ty(), builder.getPtrTy() }
            : SmallVector<Type*> { builder.getPtrTy(),
                builder.getInt64Ty(), builder.getInt64Ty(), builder.getPtrTy() };

    auto* funcType = FunctionType::get(builder.getVoidTy(), argType, false);
    func = Function::Create(funcType, Function::ExternalLinkage, funcName, getModule());

    if (_config.ampFormat == AmpFormat::Sep) {
        pReArg = func->getArg(0);      pReArg->setName("pRe");
        pImArg = func->getArg(1);      pImArg->setName("pIm");
        ctrBeginArg = func->getArg(2); ctrBeginArg->setName("ctr.begin");
        ctrEndArg = func->getArg(3);   ctrEndArg->setName("ctr.end");
        pMatArg = func->getArg(4);     pMatArg->setName("pmat");
    } else {
        pSvArg = func->getArg(0);      pSvArg->setName("pSv");
        ctrBeginArg = func->getArg(1); ctrBeginArg->setName("ctr.begin");
        ctrEndArg = func->getArg(2);   ctrEndArg->setName("ctr.end");
        pMatArg = func->getArg(3);     pMatArg->setName("pmat");
    }
    } /* end of function declaration */

    // init basic blocks
    BasicBlock* entryBB = BasicBlock::Create(*_context, "entry", func);
    BasicBlock* loopBB = BasicBlock::Create(*_context, "loop", func);
    BasicBlock* loopBodyBB = BasicBlock::Create(*_context, "loop.body", func);
    BasicBlock* retBB = BasicBlock::Create(*_context, "ret", func);

    std::vector<matrix_data_t> matrix(K * K);
    const auto loadMatrixF = [&]() {
        // set up matrix flags
        std::vector<std::pair<double, int>> uniqueEntries;
        std::vector<int> uniqueEntryIndices(2 * matrix.size());
        for (auto& m : matrix) {
            m.realLoadNeg = false;
            m.imagLoadNeg = false;
        }

        double zeroSkipThres = _config.zeroSkipThres / K;
        double shareMatrixElemThres = _config.shareMatrixElemThres / K;
        for (unsigned i = 0; i < matrix.size(); i++) {
            if (_config.forceDenseKernel) {
                matrix[i].realFlag = 2;
                matrix[i].imagFlag = 2;
                continue;
            }
            auto real = gate.gateMatrix.matrix.constantMatrix.data.at(i).real();
            auto imag = gate.gateMatrix.matrix.constantMatrix.data.at(i).imag();

            if (std::abs(real) < zeroSkipThres)
                matrix[i].realFlag = 0;
            else if (std::abs(real - 1.0) < zeroSkipThres)
                matrix[i].realFlag = 1;
            else if (std::abs(real + 1.0) < zeroSkipThres)
                matrix[i].realFlag = -1;
            else
                matrix[i].realFlag = 2;

            if (std::abs(imag) < zeroSkipThres)
                matrix[i].imagFlag = 0;
            else if (std::abs(imag - 1.0) < zeroSkipThres)
                matrix[i].imagFlag = 1;
            else if (std::abs(imag + 1.0) < zeroSkipThres)
                matrix[i].imagFlag = -1;
            else
                matrix[i].imagFlag = 2;

            if (shareMatrixElemThres > 0.0) {
                auto realIt = std::find_if(uniqueEntries.begin(), uniqueEntries.end(),
                        [thres=shareMatrixElemThres, real=real](const std::pair<double, int> pair)
                            { return std::abs(pair.first - real) < thres || std::abs(pair.first + real) < thres; });
                if (realIt == uniqueEntries.end()) {
                    uniqueEntries.push_back(std::make_pair(real, 2*i));
                    uniqueEntryIndices[2*i] = 2*i;
                } else {
                    uniqueEntryIndices[2*i] = realIt->second;
                    if (std::abs(realIt->first + real) < shareMatrixElemThres)
                        matrix[i].realLoadNeg = true;
                }
                auto imagIt = std::find_if(uniqueEntries.begin(), uniqueEntries.end(),
                        [thres=shareMatrixElemThres, imag=imag](const std::pair<double, int> pair)
                            { return std::abs(pair.first - imag) < thres || std::abs(pair.first + imag) < thres; });
                if (imagIt == uniqueEntries.end()) {
                    uniqueEntries.push_back(std::make_pair(imag, 2*i + 1));
                    uniqueEntryIndices[2*i + 1] = 2*i + 1;
                } else {
                    uniqueEntryIndices[2*i + 1] = realIt->second;
                    if (std::abs(imagIt->first + imag) < shareMatrixElemThres)
                        matrix[i].imagLoadNeg = true;
                }
            }
        }

        if (_config.loadVectorMatrix) {
            auto* matV = builder.CreateLoad(
                    VectorType::get(scalarTy, 2*K*K, false), pMatArg, "matrix");
            for (unsigned i = 0; i < matrix.size(); i++) {
                matrix[i].realVal = builder.CreateShuffleVector(
                    matV, std::vector<int>(S, 2*i), "mRe." + std::to_string(i));
                matrix[i].imagVal = builder.CreateShuffleVector(
                    matV, std::vector<int>(S, 2*i+1), "mIm." + std::to_string(i));
            }
        } else {
            for (unsigned i = 0; i < matrix.size(); i++) {
                uint64_t reLoadPosition = (shareMatrixElemThres > 0.0) ? uniqueEntryIndices[2*i] : 2ULL * i;
                auto* pReVal = builder.CreateConstGEP1_64(scalarTy, pMatArg, reLoadPosition, "pm.re." + std::to_string(i));
                auto* mReVal = builder.CreateLoad(scalarTy, pReVal, "m.re." + std::to_string(i) + ".tmp");
                matrix[i].realVal = builder.CreateVectorSplat(S, mReVal, "m.re." + std::to_string(i));

                uint64_t imLoadPosition = (shareMatrixElemThres > 0.0) ? uniqueEntryIndices[2*i+1] : 2ULL * i + 1;
                auto* pImVal = builder.CreateConstGEP1_64(scalarTy, pMatArg, imLoadPosition, "pmIm_" + std::to_string(i));
                auto* mImVal = builder.CreateLoad(scalarTy, pImVal, "m.re." + std::to_string(i) + ".tmp");
                matrix[i].imagVal = builder.CreateVectorSplat(S, mImVal, "m.re." + std::to_string(i));
            }
        }

        if (debugLevel > 1)
            std::cerr << CYAN_FG << "-- matrix loading done\n" << RESET;
    };

    // split qubits
    unsigned sepBit;
    std::vector<int> simdQubits, higherQubits, lowerQubits;
    { /* split qubits */
    unsigned _q = 0;
    auto qubitsIt = gate.qubits.cbegin();
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
    if (s == 0)
        sepBit = 0;
    else {
        sepBit = simdQubits.back();
        if (!lowerQubits.empty() && lowerQubits.back() > sepBit)
            sepBit = lowerQubits.back();
        sepBit++; // separation bit = lk + s
    }
    }

    const unsigned vecSize = 1U << sepBit;
    const unsigned vecSizex2 = vecSize << 1;
    auto* vecType = VectorType::get(scalarTy, vecSize, false);
    auto* vecTypex2 = VectorType::get(scalarTy, vecSizex2, false);

    const unsigned lk = lowerQubits.size();
    const unsigned LK = 1 << lk;
    const unsigned hk = higherQubits.size();
    const unsigned HK = 1 << hk;

    // debug print qubit splits
    if (debugLevel > 1) {
        std::cerr << CYAN_FG << "-- qubit split done\n" << RESET;
        std::cerr << "simd qubits: ";
        printVector(simdQubits) << "\n";
        std::cerr << "lower qubits: ";
        printVector(lowerQubits) << "\n";
        std::cerr << "higher qubits: ";
        printVector(higherQubits) << "\n";
        std::cerr << "sepBit:  " << sepBit << "\n";
        std::cerr << "vecSize: " << vecSize << "\n";
    }

    builder.SetInsertPoint(entryBB);
    if (_config.loadMatrixInEntry)
        loadMatrixF();
    builder.CreateBr(loopBB);

    // loop entry
    builder.SetInsertPoint(loopBB);
    PHINode* counterV = builder.CreatePHI(builder.getInt64Ty(), 2, "counter");
    counterV->addIncoming(ctrBeginArg, entryBB);
    Value* cond = builder.CreateICmpSLT(counterV, ctrEndArg, "cond");
    builder.CreateCondBr(cond, loopBodyBB, retBB);

    // loop body
    builder.SetInsertPoint(loopBodyBB);
    if (!_config.loadMatrixInEntry)
        loadMatrixF();

    // debug print matrix
    if (debugLevel > 1) {
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

    Value* idxStartV = counterV;
    { /* locate start pointer */
    if (_config.usePDEP) {
        uint64_t pdepMask = ~static_cast<uint64_t>(0ULL);
        for (unsigned hi = 0; hi < hk; hi++) {
            auto bit = higherQubits[hi];
            pdepMask ^= (1 << (bit - sepBit));
        }
        if (debugLevel > 2)
            std::cerr << "pdepMask = " << std::bitset<12>(pdepMask) << "\n";
        idxStartV = builder.CreateIntrinsic(
                idxStartV->getType(), Intrinsic::x86_bmi_pdep_64,
                { counterV, builder.getInt64(pdepMask) }, nullptr, "idxStart");
    }
    else if (!higherQubits.empty()) {
        // idx = insert 0 to every bit in higherQubits to counter
        if (debugLevel > 2)
            std::cerr << "finding masks... idxStart = sum of ((counter & mask) << i)\n";

        idxStartV = builder.getInt64(0ULL);
        uint64_t mask = 0ULL;
        uint64_t maskSum = 0ULL;
        Value* tmpCounterV = counterV;
        for (unsigned i = 0; i < higherQubits.size(); i++) {
            unsigned bit = higherQubits[i];
            mask = ((1ULL << (bit - sepBit - i)) - 1) - maskSum;
            maskSum = (1ULL << (bit - sepBit - i)) - 1;
            if (debugLevel > 2) {
                std::cerr << "i = " << i << ", bit = " << bit
                          << ", mask = " << mask << " 0b" << std::bitset<12>(mask) << "\n";
            }
            tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
            tmpCounterV = builder.CreateShl(tmpCounterV, i, "tmpCounter");
            idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "tmpIdx");
        }
        mask = ~((1ULL << (higherQubits.back() - sepBit - higherQubits.size() + 1)) - 1);
        if (debugLevel > 2) {
            std::cerr << "i = " << hk << "           mask = " << mask << " 0b"
                      << std::bitset<12>(mask) << "\n";
        }
        tmpCounterV = builder.CreateAnd(counterV, mask, "tmpCounter");
        tmpCounterV = builder.CreateShl(tmpCounterV, higherQubits.size(), "tmpCounter");
        idxStartV = builder.CreateAdd(idxStartV, tmpCounterV, "idxStart");
    }
    }

    std::vector<std::vector<int>> splits(LK);
    std::vector<int> reSplitMask, imSplitMask;
    std::vector<std::vector<int>> mergeMasks;
    std::vector<int> svMargeMask(vecSizex2);
    { /* initialize loading and storing masks */
    // loading (split) masks
    for (unsigned i = 0; i < vecSize; i++) {
        unsigned key = 0;
        for (unsigned lowerI = 0; lowerI < lk; lowerI++) {
            if (i & (1 << lowerQubits[lowerI]))
                key |= (1 << lowerI);
        }
        splits[key].push_back(i);
    }
    for (unsigned i = 0; i < vecSizex2; i++) {
        if (i & S)
            imSplitMask.push_back(i);
        else
            reSplitMask.push_back(i);
    }
    if (debugLevel > 1) {
        std::cerr << CYAN_FG << "-- loading (split) masks prepared\n" << RESET;
        std::cerr << "splits: [";
        for (const auto& split : splits) {
            std::cerr << "\n ";
            printVector(split);
        }
        std::cerr << " ]\n";
    }
    }

    /* load amplitude registers
      There are a total of 2K registers, among which K are real and K are imag
      In Alt Format, we load HK size-(2*S*LK) LLVM registers.
      There are two stages of shuffling (splits)
      - Stage 1:
        Each size-(2*S*LK) reg is shuffled into 2 size-(S*LK) regs, the real and
        imag parts.
      - Stage 2:
        Each size-(S*LK) res is shuffled into LK size-S regs, the amplitude
        vectors.
    */
    std::vector<Value*> real(K, nullptr), imag(K, nullptr);
    std::vector<Value*> pSv(HK, nullptr), pRe(HK, nullptr), pIm(HK, nullptr);
    { /* load amplitude registers */
    Value *svFull, *reFull, *imFull;
    for (unsigned hi = 0; hi < HK; hi++) {
        unsigned keyStart = 0;
        uint64_t idxShift = 0ULL;
        for (unsigned higherI = 0; higherI < hk; higherI++) {
            if (hi & (1 << higherI)) {
                idxShift += 1ULL << (higherQubits[higherI] - sepBit);
                keyStart += 1 << (higherI + lowerQubits.size());
            }
        }
        if (debugLevel > 2)
            std::cerr << "hi = " << hi << ", "
                      << "idxShift = " << std::bitset<12>(idxShift) << ", "
                      << "keyStart = " << keyStart << "\n";

        auto* _idxV = builder.CreateAdd(idxStartV, builder.getInt64(idxShift), "idx_" + std::to_string(hi));
        if (_config.ampFormat == AmpFormat::Sep) {
            pRe[hi] = builder.CreateGEP(vecType, pReArg, _idxV, "pRe." + std::to_string(hi));
            pIm[hi] = builder.CreateGEP(vecType, pImArg, _idxV, "pIm." + std::to_string(hi));
            reFull = builder.CreateLoad(vecType, pRe[hi], "re.full." + std::to_string(hi));
            imFull = builder.CreateLoad(vecType, pIm[hi], "im.full." + std::to_string(hi));
        } else {
            pSv[hi] = builder.CreateGEP(vecTypex2, pSvArg, _idxV, "pSv." + std::to_string(hi));
            svFull = builder.CreateLoad(vecTypex2, pSv[hi], "sv.full." + std::to_string(hi));
            reFull = builder.CreateShuffleVector(svFull, reSplitMask, "re.full." + std::to_string(hi));
            imFull = builder.CreateShuffleVector(svFull, imSplitMask, "im.full." + std::to_string(hi));
        }
        for (unsigned i = 0; i < LK; i++) {
            unsigned key = keyStart + i;
            real[key] = builder.CreateShuffleVector(
                reFull, splits[i], "real." + std::to_string(key));
            imag[key] = builder.CreateShuffleVector(
                imFull, splits[i], "imag." + std::to_string(key));
        }
    }
    }

    { /* prepare merge masks (override 'splits' variable) */
    // storing (merge) masks
    for (unsigned round = 0; round < lk; round++) {
        for (unsigned pairI = 0; pairI < (1 << (lk - round - 1)); pairI++) {
            auto pair = getMaskToMerge(splits[2*pairI], splits[2*pairI + 1]);
            mergeMasks.push_back(std::move(pair.first));
            splits[pairI] = std::move(pair.second);
        }
    }
    int reCount = 0, imCount = 0;
    for (unsigned i = 0; i < vecSizex2; i++) {
        if (i & S)
            svMargeMask[i] = (imCount++) + vecSize;
        else
            svMargeMask[i] = reCount++;
    }
    if (debugLevel > 1)
        std::cerr << CYAN_FG << "-- merge masks done\n" << RESET;
    }

    // mat-vec mul and storing
    for (unsigned hi = 0; hi < HK; hi++) {
        // matrix-vector multiplication
        std::vector<Value*> newReal(LK, nullptr);
        std::vector<Value*> newImag(LK, nullptr);
        for (unsigned li = 0; li < LK; li++) {
            unsigned r = hi * LK + li; // row
            // real part
            std::string nameRe = "re.new." + std::to_string(r) + ".";
            if (_config.useFMS) {
                for (unsigned c = 0; c < K; c++) {
                    if (matrix[r*K + c].realLoadNeg)
                        newReal[li] = genMulSub(newReal[li], matrix[r*K + c].realVal, real[c],
                                        matrix[r * K + c].realFlag, "", nameRe);
                    else
                        newReal[li] = genMulAdd(newReal[li], matrix[r*K + c].realVal, real[c],
                                        matrix[r * K + c].realFlag, "", nameRe);
                }
                for (unsigned c = 0; c < K; c++) {
                    if (matrix[r*K + c].imagLoadNeg)
                        newReal[li] = genMulAdd(newReal[li], matrix[r*K + c].imagVal, imag[c],
                                        matrix[r * K + c].imagFlag, "", nameRe);
                    else
                        newReal[li] = genMulSub(newReal[li], matrix[r*K + c].imagVal, imag[c],
                                        matrix[r * K + c].imagFlag, "", nameRe);
                }
            } else {
                assert(_config.shareMatrixElemThres <= 0.0 && "Not Implemented");
                Value *newRe0 = nullptr, *newRe1 = nullptr;
                for (unsigned c = 0; c < K; c++) {
                    newRe0 = genMulAdd(newRe0, matrix[r * K + c].realVal, real[c],
                                    matrix[r * K + c].realFlag, "", nameRe);
                    newRe1 = genMulAdd(newRe1, matrix[r * K + c].imagVal, imag[c],
                                    matrix[r * K + c].imagFlag, "", nameRe);
                }
                if (newRe0 != nullptr && newRe1 != nullptr)
                    newReal[li] = builder.CreateFSub(newRe0, newRe1, "newRe" + std::to_string(r) + "_");
                else if (newRe0 == nullptr)
                    newReal[li] = builder.CreateFNeg(newRe1, "newRe" + std::to_string(r) + "_");
                else if (newRe1 == nullptr)
                    newReal[li] = newRe0;
                else
                    assert(false && "newReal is null");
            }
            // imag part
            std::string nameIm = "im.new." + std::to_string(r) + ".";
            for (unsigned c = 0; c < K; c++) {
                if (matrix[r*K + c].realLoadNeg)
                    newImag[li] = genMulSub(newImag[li], matrix[r*K + c].realVal, imag[c],
                                    matrix[r * K + c].realFlag, "", nameIm);
                else
                    newImag[li] = genMulAdd(newImag[li], matrix[r*K + c].realVal, imag[c],
                                    matrix[r * K + c].realFlag, "", nameIm);
                if (matrix[r*K + c].imagLoadNeg)
                    newImag[li] = genMulSub(newImag[li], matrix[r*K + c].imagVal, real[c],
                                    matrix[r * K + c].imagFlag, "", nameIm);
                else
                    newImag[li] = genMulAdd(newImag[li], matrix[r*K + c].imagVal, real[c],
                                    matrix[r * K + c].imagFlag, "", nameIm);
            }
        }
        // merge
        if (debugLevel > 2)
            std::cerr << "Ready to merge. hi = " << hi << "\n";
        auto maskIt = mergeMasks.cbegin();
        for (unsigned round = 0; round < lk; round++) {
            if (debugLevel > 2)
                std::cerr << " round " << round
                        << ", there are " << (1 << (lk - round - 1)) << " pairs\n";
            for (unsigned pairI = 0; pairI < (1 << (lk - round - 1)); pairI++) {
                unsigned mergeIdx = 2 * pairI;
                unsigned storeIdx = pairI;
                if (debugLevel > 2)
                    std::cerr << "  pair " << pairI << ", "
                            << "(" << mergeIdx << "," << mergeIdx + 1 << ")"
                            << " => " << storeIdx << "\n";

                newReal[storeIdx] = builder.CreateShuffleVector(
                        newReal[mergeIdx], newReal[mergeIdx + 1],
                        *maskIt, "re.merge");
                newImag[storeIdx] = builder.CreateShuffleVector(
                        newImag[mergeIdx], newImag[mergeIdx + 1],
                        *maskIt, "im.merge");
                maskIt++;
            }
        }
        if (debugLevel > 2)
            std::cerr << "Merged hi = " << hi << "\n";

        // store
        if (_config.ampFormat == AmpFormat::Sep) {
            builder.CreateStore(newReal[0], pRe[hi], false);
            builder.CreateStore(newImag[0], pIm[hi], false);
        } else {
            auto* newSv = builder.CreateShuffleVector(newReal[0], newImag[0],
                                                      svMargeMask, "sv.new");
            builder.CreateStore(newSv, pSv[hi], false);
        }
    }

    // increment counter and return
    auto* counterNextV = builder.CreateAdd(counterV, builder.getInt64(1), "counter.next");
    counterV->addIncoming(counterNextV, loopBodyBB);
    builder.CreateBr(loopBB);

    builder.SetInsertPoint(retBB);
    builder.CreateRetVoid();

    return func;
}
