#include "quench/QuantumGate.h"
#include "utils/iocolor.h"

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

GateMatrix::GateMatrix(std::initializer_list<Complex<double>> m) {
    std::cerr << "Constructing ConstantMatrix\n";
    auto mSize = m.size();
    matrix = matrix_t::c_matrix_t(m);

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

std::ostream& GateMatrix::printMatrix(std::ostream& os) const {     
    assert(matrix.activeType == matrix_t::ActiveMatrixType::C
            && "Only supporting constant matrices now");

    const auto& data = matrix.constantMatrix.data;
    for (size_t r = 0; r < N; r++) {
        for (size_t c = 0; c < N; c++) {
            auto re = data[r*N + c].real;
            auto im = data[r*N + c].imag;
            if (re >= 0)
                os << " ";
            os << re;
            if (im >= 0)
                os << " + " << im << "i, ";
            else
                os << " - " << -im << "i, ";
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
        [&qubits=this->qubits](unsigned i, unsigned j) { return qubits[i] < qubits[j]; });

    std::vector<unsigned> newQubits(nqubits);
    for (unsigned i = 0; i < nqubits; i++)
        newQubits[i] = qubits[indices[i]];
    
    qubits = std::move(newQubits);
    matrix.permuteSelf(indices);
}

QuantumGate& QuantumGate::leftMatmulInplace(const QuantumGate& other) {

    return *this;
}

