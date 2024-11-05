#include "saot/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <iomanip>
#include <cmath>
#include <bitset>

using namespace saot;
using namespace IOColor;

std::ostream& QuantumGate::displayInfo(std::ostream& os) const {
    os << CYAN_FG << "QuantumGate Info\n" << RESET
       << "- Target Qubits ";
    utils::printVector(qubits) << "\n";
    os << "- Matrix:\n";
    return gateMatrix.printMatrix(os);
}

void QuantumGate::sortQubits() {
    const auto nqubits = qubits.size();
    std::vector<int> indices(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
        indices[i] = i;
    
    std::sort(indices.begin(), indices.end(),
        [&qubits=this->qubits](int i, int j) {
            return qubits[i] < qubits[j];
        });

    std::vector<int> newQubits(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
        newQubits[i] = qubits[indices[i]];
    
    qubits = std::move(newQubits);
    gateMatrix = gateMatrix.permute(indices);
}

// helper functions in QuantumGate::lmatmul
namespace {
inline uint64_t parallel_extraction(uint64_t word, uint64_t mask, int bits = 64) {
    uint64_t out = 0;
    int outIdx = 0;
    for (unsigned i = 0; i < bits; i++) {
        uint64_t ithMaskBit = (mask >> i) & 1ULL;
        uint64_t ithWordBit = (word >> i) & 1ULL;
        if (ithMaskBit == 1) {
            out |= ithWordBit << outIdx;
            outIdx += 1;
        }
    }
    return out;
}

inline QuantumGate lmatmul_up_up(
        const GateMatrix::up_matrix_t& aUp, const GateMatrix::up_matrix_t& bUp,
        const std::vector<int>& aQubits, const std::vector<int>& bQubits) {
    const int aNqubits = aQubits.size();
    const int bNqubits = bQubits.size();

    std::vector<int> cQubits;
    for (const auto& q : aQubits)
        cQubits.push_back(q);
    for (const auto& q : bQubits) {
        if (std::find(cQubits.begin(), cQubits.end(), q) == cQubits.end())
            cQubits.push_back(q);
    }
    std::sort(cQubits.begin(), cQubits.end());

    const int cNqubits = cQubits.size();
    int i = 0;
    uint64_t aMask = 0;
    uint64_t bMask = 0;
    for (auto it = cQubits.begin(); it != cQubits.end(); it++) {
        if (std::find(aQubits.begin(), aQubits.end(), *it) != aQubits.end())
            aMask |= (1 << i);
        if (std::find(bQubits.begin(), bQubits.end(), *it) != bQubits.end())
            bMask |= (1 << i);
        i++;
    }

    GateMatrix::up_matrix_t cUp(1 << cNqubits);
    for (uint64_t idx = 0; idx < (1 << cNqubits); idx++) {
        auto bIdx = parallel_extraction(idx, bMask, cQubits.back());
        auto aIdx = parallel_extraction(idx, aMask, cQubits.back());
        auto phase = bUp.data[bIdx].second + aUp.data[aIdx].second;
        cUp.data[idx] = std::make_pair(bUp.data[bIdx].first, phase);
    }
    return QuantumGate(GateMatrix(cUp), cQubits);
}
} // anonymous namespace

QuantumGate QuantumGate::lmatmul(const QuantumGate& other) const {
    // Matrix Mul A @ B = C
    // A is other, B is this
    const auto& aQubits = other.qubits;
    const auto& bQubits = qubits;
    const int aNqubits = aQubits.size();
    const int bNqubits = bQubits.size();

    struct target_qubit_t {
        int q, aIdx, bIdx;
        target_qubit_t(int q, int aIdx, int bIdx)
            : q(q), aIdx(aIdx), bIdx(bIdx) {}
    };

    // setup result gate target qubits
    std::vector<target_qubit_t> targetQubits;
    int aIdx = 0, bIdx = 0;
    while (aIdx < aNqubits || bIdx < bNqubits) {
        if (aIdx == aNqubits) {
            for (; bIdx < bNqubits; ++bIdx)
                targetQubits.emplace_back(bQubits[bIdx], -1, bIdx);
            break;
        }
        if (bIdx == bNqubits) {
            for (; aIdx < aNqubits; ++aIdx)
                targetQubits.emplace_back(aQubits[aIdx], aIdx, -1);
            break;
        }
        int aQubit = aQubits[aIdx];
        int bQubit = bQubits[bIdx];
        if (aQubit == bQubit) {
            targetQubits.emplace_back(aQubit, aIdx++, bIdx++);
            continue;
        }
        if (aQubit < bQubit) {
            targetQubits.emplace_back(aQubit, aIdx++, -1);
            continue;
        }
        targetQubits.emplace_back(bQubit, -1, bIdx++);
    }

    int cNqubits = targetQubits.size();
    uint64_t aPdepMask = 0ULL;
    uint64_t bPdepMask = 0ULL;
    uint64_t aZeroingMask = ~0ULL;
    uint64_t bZeroingMask = ~0ULL;
    std::vector<uint64_t> aSharedQubitShifts, bSharedQubitShifts;
    for (unsigned i = 0; i < cNqubits; ++i) {
        int aIdx = targetQubits[i].aIdx;
        int bIdx = targetQubits[i].bIdx;
        // shared qubit: set zeroing masks and shifts
        if (aIdx >= 0 && bIdx >= 0) {
            aZeroingMask ^= (1ULL << aIdx);
            bZeroingMask ^= (1ULL << (bIdx + bNqubits));
            aSharedQubitShifts.emplace_back(1ULL << aIdx);
            bSharedQubitShifts.emplace_back(1ULL << (bIdx + bNqubits));
        }

        if (aIdx >= 0)
            aPdepMask |= (1 << i) + (1 << (i + cNqubits));

        if (bIdx >= 0)
            bPdepMask |= (1 << i) + (1 << (i + cNqubits));

    }
    int contractionWidth = aSharedQubitShifts.size();
    std::vector<int> cQubits;
    for (const auto& tQubit : targetQubits)
        cQubits.push_back(tQubit.q);

    // std::cerr << CYAN_FG << "Debug:\n";
    // utils::printVector(aQubits, std::cerr << "aQubits: ") << "\n";
    // utils::printVector(bQubits, std::cerr << "bQubits: ") << "\n";
    // utils::printVectorWithPrinter(targetQubits, [](const target_qubit_t& q, std::ostream& os) {
    //     os << "(" << q.q << "," << q.aIdx << "," << q.bIdx << ")";
    // }, std::cerr << "target qubits: ") << "\n";
    // std::cerr << "aPdepMask: " << std::bitset<10>(aPdepMask) << "\n"
    //           << "bPdepMask: " << std::bitset<10>(bPdepMask) << "\n"
    //           << "aZeroingMask: " << std::bitset<10>(aZeroingMask) << "\n"
    //           << "bZeroingMask: " << std::bitset<10>(bZeroingMask) << "\n";
    // utils::printVector(aSharedQubitShifts, std::cerr << "a shifts: ") << "\n";
    // utils::printVector(bSharedQubitShifts, std::cerr << "b shifts: ") << "\n";
    // std::cerr << "contraction width = " << contractionWidth << "\n";
    // std::cerr << RESET;

    // unitary perm gate matrix
    // {
    // auto aUpMat = gateMatrix.getUnitaryPermMatrix();
    // auto bUpMat = other.gateMatrix.getUnitaryPermMatrix();
    // if (aUpMat.has_value() && bUpMat.has_value()) 
    //     return lmatmul_up_up(aUpMat.value(), bUpMat.value(), qubits, other.qubits);
    // }

    // const matrix
    {
    auto aOptCMat = other.gateMatrix.getConstantMatrix();
    auto bOptCMat = gateMatrix.getConstantMatrix();
    if (aOptCMat.has_value() && bOptCMat.has_value()) {
        const auto& aCMat = aOptCMat.value();
        const auto& bCMat = bOptCMat.value();
        GateMatrix::c_matrix_t cCMat(1 << cNqubits);
        // main loop
        for (uint64_t i = 0ULL; i < (1ULL << (2 * cNqubits)); i++) {
            uint64_t aIdxBegin = utils::pdep64(i, aPdepMask) & aZeroingMask;
            uint64_t bIdxBegin = utils::pdep64(i, bPdepMask) & bZeroingMask;

            // std::cerr << "Ready to update cmat[" << i << "]\n";
            // std::cerr << "  aIdxBegin: " << utils::as0b(i, 2 * cNqubits)
            //           << " -> " << utils::as0b(utils::pdep64(i, aPdepMask), 2 * aNqubits)
            //           << " -> " << utils::as0b(aIdxBegin, 2 * aNqubits) << " = " << aIdxBegin << "\n";
            // std::cerr << "  bIdxBegin: " << utils::as0b(i, 2 * cNqubits)
            //           << " -> " << utils::as0b(utils::pdep64(i, bPdepMask), 2 * bNqubits)
            //           << " -> " << utils::as0b(bIdxBegin, 2 * bNqubits) << " = " << bIdxBegin << "\n";

            for (uint64_t s = 0; s < (1ULL << contractionWidth); s++) {
                uint64_t aIdx = aIdxBegin;
                uint64_t bIdx = bIdxBegin;
                for (unsigned bit = 0; bit < contractionWidth; bit++) {
                    if (s & (1 << bit)) {
                        aIdx += aSharedQubitShifts[bit];
                        bIdx += bSharedQubitShifts[bit];
                    }
                }
                // std::cerr << "  aIdx = " << aIdx << ": " << aCMat.data[aIdx] << "; ";
                // std::cerr << "  bIdx = " << bIdx << ": " << bCMat.data[bIdx] << "\n";
                cCMat.data[i] += aCMat.data[aIdx] * bCMat.data[bIdx];
            }
        }
        return QuantumGate(GateMatrix(cCMat), cQubits);
    }
    }

    // otherwise, parametrised matrix
    auto aPMat = other.gateMatrix.getParametrizedMatrix();
    auto bPMat = gateMatrix.getParametrizedMatrix();
    GateMatrix::p_matrix_t cPMat(1 << cNqubits);
    // main loop
    for (uint64_t i = 0ULL; i < (1ULL << (2 * cNqubits)); i++) {
        uint64_t aIdxBegin = utils::pdep64(i, aPdepMask) & aZeroingMask;
        uint64_t bIdxBegin = utils::pdep64(i, bPdepMask) & bZeroingMask;

        for (uint64_t s = 0; s < (1ULL << contractionWidth); s++) {
            uint64_t aIdx = aIdxBegin;
            uint64_t bIdx = bIdxBegin;
            for (unsigned bit = 0; bit < contractionWidth; bit++) {
                if (s & (1 << bit)) {
                    aIdx += aSharedQubitShifts[bit];
                    bIdx += bSharedQubitShifts[bit];
                }
            }
            cPMat.data[i] += aPMat.data[aIdx] * bPMat.data[bIdx];
        }
    }

    return QuantumGate(GateMatrix(cPMat), cQubits);
}

// QuantumGate::opCount helper functions
namespace {
inline int opCount_c(const GateMatrix::c_matrix_t& mat, double thres) {
    int count = 0;
    for (const auto& data : mat.data) {
        if (std::abs(data.real()) >= thres)
            ++count;
        if (std::abs(data.imag()) >= thres)
            ++count;
    }
    return 2 * count;
}

inline int opCount_p(const GateMatrix::p_matrix_t& mat, double thres) {
    int count = 0;
    for (const auto& data : mat.data) {
        auto ev = data.getValue();
        if (ev.first) {
            if (std::abs(ev.second.real()) >= thres)
                ++count;
            if (std::abs(ev.second.imag()) >= thres)
                ++count;
        } else
            count += 2;
    }
    return 2 * count;
}

} // anonymous namespace

int QuantumGate::opCount(double thres) {
    if (opCountCache >= 0)
        return opCountCache;

    double normalizedThres = thres / std::pow(2.0, gateMatrix.nqubits());
    if (const auto* p = std::get_if<GateMatrix::c_matrix_t>(&gateMatrix._matrix))
        return opCountCache = opCount_c(*p, normalizedThres);
    if (const auto* p = std::get_if<GateMatrix::p_matrix_t>(&gateMatrix._matrix))
        return opCountCache = opCount_p(*p, normalizedThres);

    assert(false && "opCount Not Implemented");

    return -1;
}

// TODO: optimize it
bool QuantumGate::isConvertibleToUnitaryPermGate() const {
    return gateMatrix.getUnitaryPermMatrix().has_value();
}

bool QuantumGate::approximateGateMatrix(double thres) {
    const auto approxCplx = [thres=thres](std::complex<double>& cplx) -> bool {
        bool flag = false;
        if (std::abs(cplx.real()) < thres) {
            cplx.real(0.0);
            flag = true;
        }
        else if (std::abs(cplx.real() - 1.0) < thres) {
            cplx.real(1.0);
            flag = true;
        }
        else if (std::abs(cplx.real() + 1.0) < thres) {
            cplx.real(-1.0);
            flag = true;
        }
        
        if (std::abs(cplx.imag()) < thres) {
            cplx.imag(0.0);
            flag = true;
        }
        else if (std::abs(cplx.imag() - 1.0) < thres) {
            cplx.imag(1.0);
            flag = true;
        }
        else if (std::abs(cplx.imag() + 1.0) < thres) {
            cplx.imag(-1.0);
            flag = true;
        }
        return flag;
    };

    auto* p = std::get_if<GateMatrix::c_matrix_t>(&gateMatrix._matrix);
    if (p == nullptr)
        return false;
    bool flag = false;
    for (auto& data : p->data) {
        if (approxCplx(data))
            flag = true;
    }
    return flag;
}

void QuantumGate::simplifyGateMatrix() {
    auto* p = std::get_if<GateMatrix::p_matrix_t>(&gateMatrix._matrix);
    if (p == nullptr)
        return;
    
    for (auto& data : p->data) {
        data.removeSmallMonomials();
        data.simplifySelf();
    }
}