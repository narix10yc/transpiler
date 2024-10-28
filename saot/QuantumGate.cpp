#include "saot/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <iomanip>
#include <cmath>

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
    // Matrix Mul A @ B
    // A is other, B is this
    const int aNqubits = other.qubits.size();
    const int bNqubits = qubits.size();

    std::vector<int> allQubits;
    for (const auto& q : qubits)
        allQubits.push_back(q);
    for (const auto& q : other.qubits) {
        if (std::find(qubits.begin(), qubits.end(), q) == qubits.end())
            allQubits.push_back(q);
    }
    std::sort(allQubits.begin(), allQubits.end());

    const int newNqubits = allQubits.size();
    std::vector<uint64_t> aShift(2 * newNqubits, 0), bShift(2 * newNqubits, 0);
    std::vector<std::pair<uint64_t, uint64_t>> sShift;
    
    for (unsigned i = 0; i < newNqubits; i++) {
        const auto& q = allQubits[i];
        int aPosition = other.findQubit(q);
        int bPosition = findQubit(q);

        if (aPosition >= 0 && bPosition >= 0) {
            bShift[i] = 1ULL << bPosition; // c_q
            aShift[i+newNqubits] = 1ULL << (aPosition + aNqubits); // r_q
            sShift.push_back({1ULL << aPosition, 1ULL << (bPosition + bNqubits)}); // s_q
        } else if (aPosition >= 0) {
            aShift[i] = 1ULL << aPosition; // c_q
            aShift[i+newNqubits] = 1ULL << (aPosition + aNqubits); // r_q
        } else {
            assert(bPosition >= 0);
            bShift[i] = 1ULL << bPosition; // c_q
            bShift[i+newNqubits] = 1ULL << (bPosition + bNqubits); // r_q
        }
    }

    // std::cerr << "aShift: [";
    // for (const auto& s : aShift)
    //     std::cerr << s << ",";
    // std::cerr << "]\n" << "bShift: [";
    // for (const auto& s : bShift)
    //     std::cerr << s << ",";
    // std::cerr << "]\n" << "sShift: [";
    // for (const auto& s : sShift)
    //     std::cerr << "(" << s.first << "," << s.second << "),";
    // std::cerr << "]\n";

    // unitary perm gate matrix
    auto aUpMat = gateMatrix.getUnitaryPermMatrix();
    auto bUpMat = other.gateMatrix.getUnitaryPermMatrix();
    if (aUpMat.has_value() && bUpMat.has_value()) 
        return lmatmul_up_up(aUpMat.value(), bUpMat.value(), qubits, other.qubits);
    
    const auto twiceNewNqubits = 2 * newNqubits;
    const auto contractionBitwidth = sShift.size();
    // constant gate matrix
    auto aCMat = gateMatrix.getConstantMatrix();
    auto bCMat = other.gateMatrix.getConstantMatrix();
    if (aCMat.has_value() && bCMat.has_value()) {
        GateMatrix::c_matrix_t newCMatrix(1 << newNqubits);
        for (size_t i = 0; i < (1 << twiceNewNqubits); i++) {
            auto aPtrStart = aCMat.value().data.data();
            auto bPtrStart = bCMat.value().data.data();
            for (unsigned bit = 0; bit < twiceNewNqubits; bit++) {
                if ((i & (1 << bit)) != 0) {
                    aPtrStart += aShift[bit];
                    bPtrStart += bShift[bit];
                }
            }

            newCMatrix.data[i] = {0.0, 0.0};
            for (size_t j = 0; j < (1 << contractionBitwidth); j++) {
                auto aPtr = aPtrStart;
                auto bPtr = bPtrStart;
                for (unsigned bit = 0; bit < contractionBitwidth; bit++) {
                    if ((j & (1 << bit)) != 0) {
                        aPtr += sShift[bit].first;
                        bPtr += sShift[bit].second;
                    }
                }
                newCMatrix.data[i] += (*aPtr) * (*bPtr);
            }
        }
        return QuantumGate(GateMatrix(newCMatrix), allQubits);
    }

    // otherwise, parametrised matrix
    auto aPMat = gateMatrix.getParametrizedMatrix();
    auto bPMat = other.gateMatrix.getParametrizedMatrix();
    GateMatrix::p_matrix_t cPMat(1 << newNqubits);
    for (size_t i = 0; i < (1 << twiceNewNqubits); i++) {
        auto aPtrStart = aPMat.data.data();
        auto bPtrStart = bPMat.data.data();
        for (unsigned bit = 0; bit < twiceNewNqubits; bit++) {
            if ((i & (1 << bit)) != 0) {
                aPtrStart += aShift[bit];
                bPtrStart += bShift[bit];
            }
        }

        cPMat.data[i] = Polynomial();
        for (size_t j = 0; j < (1 << contractionBitwidth); j++) {
            auto* aPtr = aPtrStart;
            auto* bPtr = bPtrStart;
            for (unsigned bit = 0; bit < contractionBitwidth; bit++) {
                if ((j & (1 << bit)) != 0) {
                    aPtr += sShift[bit].first;
                    bPtr += sShift[bit].second;
                }
            }
            cPMat.data[i] += (*aPtr) * (*bPtr);
        }
    }
    return QuantumGate(GateMatrix(cPMat), allQubits);
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