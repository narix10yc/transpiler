#include "quench/QuantumGate.h"
#include "utils/iocolor.h"
#include <iomanip>

using namespace Color;
using namespace quench::complex_matrix;
using namespace quench::quantum_gate;

namespace {
    bool _isValidShuffleFlag(const std::vector<unsigned>& flags) {
        auto copy = flags;
        std::sort(copy.begin(), copy.end());
        for (unsigned i = 0; i < copy.size(); i++) {
            if (copy[i] != i)
                return false;
        }
        return true;
    }
}

GateMatrix GateMatrix::permute(const std::vector<unsigned>& flags) const {
    assert(nqubits == flags.size());
    assert(_isValidShuffleFlag(flags));
    assert(isConstantMatrix());

    bool isConstantShuffleFlag = true;
    for (unsigned i = 0; i < nqubits; i++) {
        if (flags[i] != i) {
            isConstantShuffleFlag = false;
            break;
        }
    }
    if (isConstantShuffleFlag)
        return *this;

    auto permuteIndex = [&flags, k=flags.size()](size_t idx) -> size_t {
        size_t newIdx = 0;
        for (unsigned b = 0; b < k; b++) {
            newIdx += ((idx & (1ULL<<b)) >> b) << flags[b];
        }
        return newIdx;
    };

    const size_t size = matrix.getSize();
    GateMatrix m;
    m.nqubits = nqubits;
    m.N = N;
    m.matrix = matrix_t::c_matrix_t(size);

    for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.matrix.constantMatrix.data[permuteIndex(r) * size + permuteIndex(c)]
                = matrix.constantMatrix.data[r * size + c];
        }
    }

    assert(m.checkConsistency());
    return m;
}

GateMatrix& GateMatrix::permuteSelf(const std::vector<unsigned>& flags) {
    assert(nqubits == flags.size());
    assert(_isValidShuffleFlag(flags));
    assert(isConstantMatrix());

    bool isConstantShuffleFlag = true;
    for (unsigned i = 0; i < nqubits; i++) {
        if (flags[i] != i) {
            isConstantShuffleFlag = false;
            break;
        }
    }
    if (isConstantShuffleFlag)
        return *this;

    auto permuteIndex = [&flags, k=flags.size()](size_t idx) -> size_t {
        size_t newIdx = 0;
        for (unsigned b = 0; b < k; b++)
            newIdx += ((idx & (1ULL<<b)) >> b) << flags[b];
        return newIdx;
    };
    const size_t size = matrix.getSize();
    matrix_t::c_matrix_t newCMatrix(size);

    for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            newCMatrix.data[permuteIndex(r) * size + permuteIndex(c)]
                = matrix.constantMatrix.data[r * size + c];
        }
    }

    matrix = std::move(newCMatrix);
    return *this;
}

int GateMatrix::updateNqubits() {
    const auto mSize = matrix.getSize();
    switch (mSize) {
    case 0:
        assert(false && "Empty matrix");
        break;
    case 1:
        assert(false && "1x1 matrix does not represent quantum gates");
        break;
    case 4: nqubits = 1; break;
    case 16: nqubits = 2; break;
    case 64: nqubits = 3; break;
    case 256: nqubits = 4; break;
    default:
        nqubits = std::log2(mSize);
        assert(std::pow(2, nqubits) == mSize);
        break;
    }
    N = 1 << nqubits;
    return nqubits;
}


GateMatrix GateMatrix::FromName(const std::string& name,
                                const std::vector<double>& params)
{
    if (name == "u3") {
        assert(params.size() == 3);
        const double theta = 0.5 * params[0];
        const auto& phi = params[1];
        const auto& lambd = params[2];
        const double ctheta = std::cos(theta);
        const double stheta = std::sin(theta);

        return { 
            {ctheta, 0},
            {-std::cos(lambd) * stheta, -std::sin(lambd) * stheta},
            {std::cos(phi) * stheta, std::sin(phi) * stheta},
            {std::cos(phi + lambd) * ctheta, std::sin(phi + lambd) * ctheta}
        };
    }
    if (name == "cx") {
        return {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {1,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0}
        };
    }

    if (name == "cz") {
        return {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {-1,0}
        };
    }

    std::cerr << RED_FG << BOLD << "Error: Unrecognized gate '" << name << "'"
              << RESET << "\n";
    return GateMatrix();
}

namespace {
    std::ostream&
    print_complex(std::ostream& os, Complex<double> c, int precision=3) {
        const double thres = 0.5 * std::pow(0.1, precision);
        if (c.real >= -thres)
            os << " ";
        if (std::fabs(c.real) < thres)
            os << "0." << std::string(precision, ' ');
        else
            os << std::fixed << std::setprecision(precision) << c.real;
        
        if (c.imag >= -thres)
            os << "+";
        if (std::fabs(c.imag) < thres)
            os << "0." << std::string(precision, ' ');
        else
            os << std::fixed << std::setprecision(precision) << c.imag;
        return os << "i";
    }
}

std::ostream& GateMatrix::printMatrix(std::ostream& os) const {     
    assert(matrix.activeType == matrix_t::ActiveMatrixType::C
            && "Only supporting constant matrices now");
            
    const auto& data = matrix.constantMatrix.data;
    for (size_t r = 0; r < N; r++) {
        for (size_t c = 0; c < N; c++) {
            print_complex(os, data[r * N + c], 3) << ", ";
        }
        os << "\n";
    }
    return os;   
}

std::ostream& QuantumGate::displayInfo(std::ostream& os) const {
    os << "QuantumGate on qubits [";
    for (const auto& q : qubits)
        os << q << ",";
    os << "]\n"
       << "Matrix:\n";
    matrix.printMatrix(os);
    return os;
}

void QuantumGate::sortQubits() {
    const auto nqubits = qubits.size();
    std::vector<unsigned> indices(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
        indices[i] = i;
    
    std::sort(indices.begin(), indices.end(),
        [&qubits=this->qubits](unsigned i, unsigned j) {
            return qubits[i] < qubits[j];
        });

    std::vector<unsigned> newQubits(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
        newQubits[i] = qubits[indices[i]];
    
    qubits = std::move(newQubits);
    matrix.permuteSelf(indices);
}

QuantumGate QuantumGate::lmatmul(const QuantumGate& other) const {
    // Matrix Mul A @ B
    // A is other, B is this
    const unsigned aNqubits = other.qubits.size();
    const unsigned bNqubits = qubits.size();

    std::vector<unsigned> allQubits;
    for (const auto& q : qubits)
        allQubits.push_back(q);
    for (const auto& q : other.qubits) {
        if (std::find(qubits.begin(), qubits.end(), q) == qubits.end())
            allQubits.push_back(q);
    }
    std::sort(allQubits.begin(), allQubits.end());

    const auto newNqubits = allQubits.size();
    std::vector<size_t> aShift(2 * newNqubits, 0), bShift(2 * newNqubits, 0);
    std::vector<std::pair<size_t, size_t>> sShift;
    
    for (unsigned i = 0; i < newNqubits; i++) {
        const auto& q = allQubits[i];
        int aPosition = other.findQubit(q);
        int bPosition = findQubit(q);

        if (aPosition >= 0 && bPosition >= 0) {
            bShift[i] = 1 << bPosition; // c_q
            aShift[i+newNqubits] = 1 << (aPosition + aNqubits); // r_q
            sShift.push_back({1 << aPosition, 1 << (bPosition + bNqubits)}); // s_q
        } else if (aPosition >= 0) {
            aShift[i] = 1 << aPosition; // c_q
            aShift[i+newNqubits] = 1 << (aPosition + aNqubits); // r_q
        } else {
            assert(bPosition >= 0);
            bShift[i] = 1 << bPosition; // c_q
            bShift[i+newNqubits] = 1 << (bPosition + bNqubits); // r_q
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

    matrix_t::c_matrix_t newCMatrix(1 << newNqubits);
    using complex_t = complex_matrix::Complex<double>;

    assert(other.matrix.isConstantMatrix());
    assert(matrix.isConstantMatrix());
    const auto twiceNewNqubits = 2 * newNqubits;
    const auto contractionBitwidth = sShift.size();
    for (size_t i = 0; i < (1 << twiceNewNqubits); i++) {
        auto aPtrStart = other.matrix.matrix.constantMatrix.data.data();
        auto bPtrStart = matrix.matrix.constantMatrix.data.data();
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

    return {newCMatrix, allQubits};
}

