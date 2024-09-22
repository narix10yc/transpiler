#include "quench/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <iomanip>
#include <cmath>

using namespace Color;
using namespace quench::complex_matrix;
using namespace quench::quantum_gate;
using namespace saot;

bool _isValidShuffleFlag(const std::vector<int>& flags) {
    auto copy = flags;
    std::sort(copy.begin(), copy.end());
    for (unsigned i = 0; i < copy.size(); i++) {
        if (copy[i] != i)
            return false;
    }
    return true;
}

GateMatrix& GateMatrix::approximateSelf(int level, double thres) {
    assert(isConstantMatrix() && "approximateSelf only works for constant matrix");
    if (level < 1)
        return *this;
    
    auto& cMat = matrix.constantMatrix;
    for (auto& cplx : cMat.data) {
        if (std::abs(cplx.real()) < thres)
            cplx.real(0.0);
        else if (level > 1) {
            if (std::abs(cplx.real() - 1.0) < thres)
                cplx.real(1.0);
            else if (std::abs(cplx.real() + 1.0) < thres)
                cplx.real(-1.0);
        }
        
        if (std::abs(cplx.imag()) < thres)
            cplx.imag(0.0);
        else if (level > 1) {
            if (std::abs(cplx.imag() - 1.0) < thres)
                cplx.imag(1.0);
            else if (std::abs(cplx.imag() + 1.0) < thres)
                cplx.imag(-1.0);
        }
    }
    return *this;
}

GateMatrix GateMatrix::permute(const std::vector<int>& flags) const {
    assert(nqubits == flags.size());
    assert(_isValidShuffleFlag(flags));

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
    if (isConstantMatrix())
        m.matrix = matrix_t::c_matrix_t(size);
    else  {
        assert(isParametrizedMatrix());
        m.matrix = matrix_t::p_matrix_t(size);
    }

    for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            auto idxNew = permuteIndex(r) * size + permuteIndex(c);
            auto idxOld = r * size + c;
            if (isConstantMatrix())
                m.matrix.constantMatrix.data[idxNew] = matrix.constantMatrix.data[idxOld];
            else
                m.matrix.parametrizedMatrix.data[idxNew] = matrix.parametrizedMatrix.data[idxOld];
        }
    }

    assert(m.checkConsistency());
    return m;
}

int GateMatrix::updateNqubits() {
    const auto mSize = matrix.getSize();
    switch (mSize) {
    case 0:
        assert(false && "Empty matrix");
        break;
    case 1:
        assert(false && "1x1 matrix does not represent quantum gates");
        nqubits = 0;
        break;
    case  2: nqubits = 1; break;
    case  4: nqubits = 2; break;
    case  8: nqubits = 3; break;
    case 16: nqubits = 4; break;
    default:
        nqubits = std::log2(mSize);
        assert(std::pow(2, nqubits) == mSize);
        break;
    }
    N = 1 << nqubits;
    return nqubits;
}

GateMatrix GateMatrix::FromName(
        const std::string& name, const std::vector<double>& params) {
    if (name == "u3" || name == "u1q") { 
        if (params.size() != 3) {
            std::cerr << "Got " << params.size() << " params\n";
        }
        assert(params.size() == 3);
        const double theta = 0.5 * params[0];
        const auto phi = params[1];
        const auto& lambd = params[2];
        const double ctheta = std::cos(theta);
        const double stheta = std::sin(theta);

        return {{
            {ctheta, 0},
            {-std::cos(lambd) * stheta, -std::sin(lambd) * stheta},
            {std::cos(phi) * stheta, std::sin(phi) * stheta},
            {std::cos(phi + lambd) * ctheta, std::sin(phi + lambd) * ctheta}
        }};
    }

    if (name == "h") {
        return {{
            { M_SQRT1_2, 0}, { M_SQRT1_2, 0},
            { M_SQRT1_2, 0}, {-M_SQRT1_2, 0} 
        }};
    }

    if (name == "cx") {
        return {{
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {1,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0}
        }};
    }

    if (name == "cz") {
        return {{
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {-1,0}
        }};
    }

    if (name == "cp") {
        assert(params.size() == 1);
        return {{
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {std::cos(params[0]), std::sin(params[0])}
        }};
    }

    std::cerr << RED_FG << BOLD << "Error: Unrecognized gate '" << name << "'"
              << RESET << "\n";
    assert(false);
    return GateMatrix();
}

GateMatrix GateMatrix::FromParameters(
        const std::string& name,
        const std::vector<GateParameter>& params) {
    if (name == "u3" || name == "u1q") {
        assert(params.size() == 3 && "u1q gate needs 3 parameters");

        const auto theta = params[0].isConstant ? -1 : params[0].variable;
        const auto lambd = params[1].isConstant ? -2 : params[1].variable;
        const auto phi   = params[2].isConstant ? -3 : params[2].variable;

        return GateMatrix(matrix_t::p_matrix_t {
            Polynomial(Monomial(std::complex<double>(1.0, 0.0), { VariableSumNode::Cosine({theta})}, {})),
            Polynomial(Monomial(std::complex<double>(-1.0, 0.0), { VariableSumNode::Sine({theta}) }, {lambd})),
            Polynomial(Monomial(std::complex<double>(1.0, 0.0), { VariableSumNode::Sine({theta}) }, { phi })),
            Polynomial(Monomial(std::complex<double>(1.0, 0.0), { VariableSumNode::Cosine({theta}) }, { phi, lambd }))
        });
    }

    if (name == "h") {
        GateMatrix gateMat(matrix_t::c_matrix_t {
            { M_SQRT1_2, 0 }, { M_SQRT1_2, 0 },
            { M_SQRT1_2, 0 }, {-M_SQRT1_2, 0 } 
        });
        gateMat.convertToParametrizedMatrix();
        return gateMat;
    }

    if (name == "cx") {
        GateMatrix gateMat(matrix_t::c_matrix_t {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {0.0}, {0.0}, {1,0},
            {0,0}, {0.0}, {1.0}, {0,0},
            {0,0}, {1.0}, {0.0}, {0,0}
        });
        gateMat.convertToParametrizedMatrix();
        return gateMat;
    }

    if (name == "cz") {
        GateMatrix gateMat(matrix_t::c_matrix_t {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {-1,0}
        });
        gateMat.convertToParametrizedMatrix();
        return gateMat;
    }
        
    std::cerr << RED_FG << BOLD << "Not Implemented gate '" << name << "'"
              << RESET << "\n";
    assert(false);
    return GateMatrix();
}

std::ostream& GateMatrix::printMatrix(std::ostream& os) const {     
    if (isConstantMatrix()) {
        const auto& data = matrix.constantMatrix.data;
        os << "[";
        for (size_t r = 0; r < N; r++) {
            for (size_t c = 0; c < N; c++) {
                utils::print_complex(os, data[r * N + c], 3);
                if (c != N-1 || r != N-1)
                    os << ",";
                os << " ";
            }
            if (r == N-1)
                os << "]\n";
            else 
                os << "\n ";
        }
        return os;
    } else {
        const auto& data = matrix.parametrizedMatrix.data;
        for (size_t r = 0; r < N; r++) {
            for (size_t c = 0; c < N; c++) {
                os << "[" << r << "," << c << "]: ";
                data[r*N + c].print(os) << "\n";
            }
        }
        return os;
    }
}

std::ostream& QuantumGate::displayInfo(std::ostream& os) const {
    os << CYAN_FG << "QuantumGate Info\n" << RESET
       << "- Target Qubits ";
    utils::printVector(qubits) << "\n";
    os << "- Matrix:\n";
    gateMatrix.printMatrix(os);
    return os;
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

    const auto twiceNewNqubits = 2 * newNqubits;
    const auto contractionBitwidth = sShift.size();

    // constant gate matrix
    if (other.gateMatrix.isConstantMatrix() && gateMatrix.isConstantMatrix()) {
        matrix_t::c_matrix_t newCMatrix(1 << newNqubits);
        for (size_t i = 0; i < (1 << twiceNewNqubits); i++) {
            auto aPtrStart = other.gateMatrix.matrix.constantMatrix.data.data();
            auto bPtrStart = gateMatrix.matrix.constantMatrix.data.data();
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
        return QuantumGate(newCMatrix, allQubits);
    }

    // otherwise, parametrised matrix
    assert(other.gateMatrix.isParametrizedMatrix());
    assert(gateMatrix.isParametrizedMatrix());
    matrix_t::p_matrix_t newPMatrix(1 << newNqubits);
    for (size_t i = 0; i < (1 << twiceNewNqubits); i++) {
        auto aPtrStart = other.gateMatrix.matrix.parametrizedMatrix.data.data();
        auto bPtrStart = gateMatrix.matrix.parametrizedMatrix.data.data();
        for (unsigned bit = 0; bit < twiceNewNqubits; bit++) {
            if ((i & (1 << bit)) != 0) {
                aPtrStart += aShift[bit];
                bPtrStart += bShift[bit];
            }
        }

        newPMatrix.data[i] = Polynomial();
        for (size_t j = 0; j < (1 << contractionBitwidth); j++) {
            auto* aPtr = aPtrStart;
            auto* bPtr = bPtrStart;
            for (unsigned bit = 0; bit < contractionBitwidth; bit++) {
                if ((j & (1 << bit)) != 0) {
                    aPtr += sShift[bit].first;
                    bPtr += sShift[bit].second;
                }
            }
            newPMatrix.data[i] += (*aPtr) * (*bPtr);
        }
    }
    return QuantumGate(newPMatrix, allQubits);
}

int QuantumGate::opCount(double thres) {
    if (opCountCache >= 0)
        return opCountCache;

    int count = 0;
    double normalizedThres = thres / std::pow(2.0, gateMatrix.nqubits);
    if (gateMatrix.isConstantMatrix()) {
        for (const auto& data : gateMatrix.matrix.constantMatrix.data) {
            if (std::abs(data.real()) >= normalizedThres)
                count++;
            if (std::abs(data.imag()) >= normalizedThres)
                count++;
        }
        opCountCache = 2 * count;
        return opCountCache;
    }
    else {
        assert(gateMatrix.isParametrizedMatrix());
        for (const auto& data : gateMatrix.matrix.parametrizedMatrix.data) {
            auto ev = data.getValue();
            if (ev.first) {
                if (std::abs(ev.second.real()) >= normalizedThres)
                    count++;
                if (std::abs(ev.second.imag()) >= normalizedThres)
                    count++;
            } else {
                count += 2;
            }
        }
        opCountCache = 2 * count;
        return opCountCache;
    }
    assert(false && "Unreachable");

    return -1;
}
