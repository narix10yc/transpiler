#include "saot/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <iomanip>
#include <cmath>
#include <type_traits>

using namespace IOColor;
using namespace saot;

GateKind saot::String2GateKind(const std::string& s) {
    if (s == "x") return gX;
    if (s == "y") return gY;
    if (s == "z") return gZ;
    if (s == "h") return gH;
    if (s == "u3" || s == "u1q") return gU;
    if (s == "cx") return gCX;
    if (s == "cz") return gCZ;
    if (s == "cp") return gCP;

    assert(false && "Unimplemented String2GateKind");
    return gUndef;
}

std::string saot::GateKind2String(GateKind t) {
    switch (t) {
    case gX: return "x";
    case gY: return "y";
    case gZ: return "z";
    case gH: return "h";
    case gU: return "u1q";

    case gCX: return "cx";
    case gCZ: return "cz";
    case gCP: return "cp";
    
    default:
        assert(false && "Unimplemented GateKind2String");
        return "undef";
    }
}

using p_matrix_t = GateMatrix::p_matrix_t;
using up_matrix_t = GateMatrix::up_matrix_t;
using c_matrix_t = GateMatrix::c_matrix_t;
using gate_params_t = GateMatrix::gate_params_t;

#pragma region Static UP Matrices
const up_matrix_t GateMatrix::MatrixI1_up {
    {0, 0.0}, {1, 0.0}
};

const up_matrix_t GateMatrix::MatrixI2_up {
    {0, 0.0}, {1, 0.0}, {2, 0.0}, {3, 0.0}
};

const up_matrix_t GateMatrix::MatrixX_up = up_matrix_t({
    {1, 0.0}, {0, 0.0}
});

const up_matrix_t GateMatrix::MatrixY_up = up_matrix_t({
    {1, -M_PI_2}, {0, M_PI}
});

const up_matrix_t GateMatrix::MatrixZ_up = up_matrix_t({
    {0, 0.0}, {1, M_PI}
});

const up_matrix_t GateMatrix::MatrixCX_up = up_matrix_t({
    {0, 0.0}, {3, 0.0}, {2, 0.0}, {1, 0.0}
});

const up_matrix_t GateMatrix::MatrixCZ_up = up_matrix_t({
    {0, 0.0}, {1, 0.0}, {2, 0.0}, {3, M_PI}
});

#pragma endregion

#pragma region Static Constant Matrices
const c_matrix_t GateMatrix::MatrixI1_c = {
    {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}
};

const c_matrix_t GateMatrix::MatrixI2_c = c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
});

const c_matrix_t GateMatrix::MatrixX_c = c_matrix_t({
    {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
});

const c_matrix_t GateMatrix::MatrixY_c({
    {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}
});

const c_matrix_t GateMatrix::MatrixZ_c = c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
});

const c_matrix_t GateMatrix::MatrixH_c = c_matrix_t({
    {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}
});


const c_matrix_t GateMatrix::MatrixCX_c = c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
});

const c_matrix_t GateMatrix::MatrixCZ_c = c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},
});

#pragma endregion

#pragma region Static Parametrized Matrices
const p_matrix_t GateMatrix::MatrixI1_p = {
    {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}
};

const p_matrix_t GateMatrix::MatrixI2_p = {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
};

const p_matrix_t GateMatrix::MatrixX_p = p_matrix_t({
    {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
});

const p_matrix_t GateMatrix::MatrixY_p({
    {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}
});

const p_matrix_t GateMatrix::MatrixZ_p = p_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
});

const p_matrix_t GateMatrix::MatrixH_p = p_matrix_t({
    {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}
});


const p_matrix_t GateMatrix::MatrixCX_p {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
};

const p_matrix_t GateMatrix::MatrixCZ_p = p_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},
});

#pragma endregion

int getNumActiveParams(const gate_params_t& params) {
    for (int i = 0; i < params.size(); i++) {
        if (params[i].index() == 0)
            return i;
    }
    return params.size();
}

GateMatrix GateMatrix::FromName(const std::string& name, const gate_params_t& params) {
    if (name == "x") {
        assert(getNumActiveParams(params) == 0 && "X gate has 0 parameter");
        return GateMatrix(gX);
    }
    if (name == "y") {
        assert(getNumActiveParams(params) == 0 && "Y gate has 0 parameter");
        return GateMatrix(gY);
    }
    if (name == "z") {
        assert(getNumActiveParams(params) == 0 && "Z gate has 0 parameter");
        return GateMatrix(gZ);
    }
    if (name == "h") {
        assert(getNumActiveParams(params) == 0 && "H gate has 0 parameter");
        return GateMatrix(gH);
    }
    if (name == "p") {
        assert(getNumActiveParams(params) == 1 && "P gate has 1 parameter");
        return GateMatrix(gP, params);
    }
    if (name == "rx") {
        assert(getNumActiveParams(params) == 1 && "RX gate has 1 parameter");
        return GateMatrix(gRX, params);
    }
    if (name == "ry") {
        assert(getNumActiveParams(params) == 1 && "RY gate has 1 parameter");
        return GateMatrix(gRY, params);
    }
    if (name == "rz") {
        assert(getNumActiveParams(params) == 1 && "RZ gate has 1 parameter");
        return GateMatrix(gRZ, params);
    }
    if (name == "u3" || name == "u1q") {
        assert(getNumActiveParams(params) == 3 && "U3 (U1q) gate has 3 parameters");
        return GateMatrix(gU, params);
    }

    if (name == "cx") {
        assert(getNumActiveParams(params) == 0 && "CX gate has 0 parameter");
        return GateMatrix(gCX);
    }
    if (name == "cz") {
        assert(getNumActiveParams(params) == 0 && "CZ gate has 0 parameter");
        return GateMatrix(gCZ);
    }
    if (name == "cp") {
        assert(getNumActiveParams(params) == 1 && "CP gate has 1 parameter");
        return GateMatrix(gCP, params);
    }

    assert(false && "Unsupported gate");
    return GateMatrix(gUndef);
}


namespace { // GateMatrix::nqubits definition
inline int nqubits_params(GateKind ty) {
    if (ty >= 1)
        return ty;
    switch (ty) {
        case gX:  return 1;
        case gY:  return 1;
        case gZ:  return 1;
        case gH:  return 1;
        case gP:  return 1;
        case gRX: return 1;
        case gRY: return 1;
        case gRZ: return 1;
        case gU:  return 1;
        
        case gCX: return 2;
        case gCZ: return 2;
        case gSWAP: return 2;
        case gCP: return 2;
        
        default:
            assert(false && "Unknown GateKind nqubits");
            return 0;
    }
}

inline int nqubits_up(const up_matrix_t& matrix) {
    int nqubits = std::log2(matrix.data.size());
    assert((1 << nqubits) == matrix.data.size());
    return nqubits;
}

inline int nqubits_c(const c_matrix_t& matrix) {
    int nqubits = std::log2(matrix.edgeSize());
    assert(1 << nqubits == matrix.edgeSize());
    return nqubits;
}

inline int nqubits_p(const p_matrix_t& matrix) {
    int nqubits = std::log2(matrix.edgeSize());
    assert(1 << nqubits == matrix.edgeSize());
    return nqubits;
}
} // anonymous namespace

int GateMatrix::nqubits() const {
    if (const auto* p = std::get_if<gate_params_t>(&_matrix))
        return nqubits_params(gateKind);
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return nqubits_up(*p);
    if (const auto* p = std::get_if<c_matrix_t>(&_matrix))
        return nqubits_c(*p);
    if (const auto* p = std::get_if<p_matrix_t>(&_matrix))
        return nqubits_p(*p);
    assert(false && "Unknown matrix type");
    return 0;
}

GateMatrix& GateMatrix::approximateSelf(int level, double thres) {
    if constexpr (std::is_same_v<decltype(_matrix), c_matrix_t>) {
        assert(false && "Should only call approximateSelf on constant gate matrices");
        return *this;
    }

    if (level < 1)
        return *this;
    
    auto& cMat = std::get<c_matrix_t>(_matrix);
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

inline GateMatrix
permute_params(const GateMatrix& gateMatrix, const std::vector<int>& flags) {
    return gateMatrix;
}

GateMatrix GateMatrix::permute(const std::vector<int>& flags) const {
    if (const auto* p = std::get_if<gate_params_t>(&_matrix))
        return permute_params(*this, flags);
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return GateMatrix(p->permute(flags));
    if (const auto* p = std::get_if<c_matrix_t>(&_matrix))
        return GateMatrix(p->permute(flags));
    if (const auto* p = std::get_if<p_matrix_t>(&_matrix))
        return GateMatrix(p->permute(flags));

    assert(false && "Unknown matrix type");
    return *this;
}

namespace { // GateMatrix::printMatrix definition
inline std::ostream&
printMatrix_params(std::ostream& os, const GateMatrix&) {

    return os << "print params_t matrix\n";
}

inline std::ostream&
printMatrix_up(std::ostream& os, const up_matrix_t&) {

    return os << "print up_t matrix\n";
}

inline std::ostream&
printMatrix_c(std::ostream& os, const c_matrix_t& matrix) {
    const auto& data = matrix.data;
    auto edgeSize = matrix.edgeSize();
    os << "[";
    for (size_t r = 0; r < edgeSize; r++) {
        for (size_t c = 0; c < edgeSize; c++) {
            utils::print_complex(os, data[r * edgeSize + c], 3);
            if (c != edgeSize-1 || r != edgeSize-1)
                os << ",";
            os << " ";
        }
        if (r == edgeSize-1)
            os << "]\n";
        else 
            os << "\n ";
    }
    return os;
}

inline std::ostream&
printMatrix_p(std::ostream& os, const p_matrix_t& matrix) {
    auto edgeSize = matrix.edgeSize();
    const auto& data = matrix.data;
    for (size_t r = 0; r < edgeSize; r++) {
        for (size_t c = 0; c < edgeSize; c++) {
            os << "[" << r << "," << c << "]: ";
            data[r*edgeSize + c].print(os) << "\n";
        }
    }
    return os;
}
} // anonymous namespace

std::ostream& GateMatrix::printMatrix(std::ostream& os) const {     
    return std::visit([this, &os](auto&& arg) -> std::ostream& {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, gate_params_t>)
            return printMatrix_params(os, *this);
        if constexpr (std::is_same_v<T, up_matrix_t>)
            return printMatrix_up(os, arg);
        if constexpr (std::is_same_v<T, c_matrix_t>)
            return printMatrix_c(os, arg);
        if constexpr (std::is_same_v<T, p_matrix_t>)
            return printMatrix_p(os, arg);
        assert(false && "Unknown matrix type");
        return os;
    }, _matrix);
}

std::ostream& GateMatrix::printParametrizedMatrix(std::ostream& os) const {
    return printMatrix_p(os, getParametrizedMatrix());
}

std::optional<up_matrix_t> GateMatrix::getUnitaryPermMatrix() const {
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return *p;

    if (const auto* p = std::get_if<gate_params_t>(&_matrix)) {
        switch (gateKind) {
        case gX: return MatrixX_up;
        case gY: return MatrixY_up;
        case gZ: return MatrixZ_up;
        case gP: {
            const double* plambd = std::get_if<double>(&(*p)[0]);
            assert(plambd && "");
            return up_matrix_t {{0, 0.0}, {1, *plambd}};
        }
        case gCX: return MatrixCX_up;
        case gCZ: return MatrixCZ_up;
        case gCP: {
            const double* plambd = std::get_if<double>(&(*p)[0]);
            assert(plambd);
            return up_matrix_t {{0, 0.0}, {0, 0.0}, {0, 0.0}, {1, *plambd}};
        }
        default:
            break;
        }
    }

    if (const auto* p = std::get_if<c_matrix_t>(&_matrix)) {
        const auto edgeSize = p->edgeSize();
        up_matrix_t upMat(edgeSize);
        for (size_t r = 0; r < edgeSize; r++) {
            bool rowFlag = false;
            for (size_t c = 0; c < edgeSize; c++) {
                const auto& cplx = p->data[r * edgeSize + c];
                if (cplx != std::complex<double>{ 0.0, 0.0 }) {
                    if (rowFlag)
                        return std::nullopt;
                    rowFlag = true;
                    upMat.data[r] = { c, std::atan2(cplx.imag(), cplx.real()) };
                }
            }
            if (!rowFlag)
                return std::nullopt;
        }
        return upMat;
    }
    
    return std::nullopt;
}

// GateMatrix::getConstantMatrix definition
namespace {
inline c_matrix_t getMatrixU_c(double theta, double phi, double lambd) {
    double ctheta = std::cos(theta);
    double stheta = std::sin(theta);
    return c_matrix_t({
        { ctheta, 0.0 },
        { -std::cos(lambd) * stheta, -std::sin(lambd) * stheta },
        { std::cos(phi) * stheta, std::sin(phi) * stheta },
        { std::cos(phi+lambd) * ctheta, std::sin(phi+lambd) * ctheta }
    });
}

inline std::optional<c_matrix_t>
getConstantMatrix_params(GateKind ty, const gate_params_t& params) {
    switch (ty) {
    case gX: return GateMatrix::MatrixX_c;
    case gY: return GateMatrix::MatrixY_c;
    case gZ: return GateMatrix::MatrixZ_c;
    case gH: return GateMatrix::MatrixH_c;
    case gP: {
        const double* p = std::get_if<double>(&params[0]);
        if (p)
            return c_matrix_t({
                { 1.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { std::cos(*p), std::sin(*p) }
            });
        return std::nullopt;
    }

    case gU: {
        const double* p0 = std::get_if<double>(&params[0]);
        const double* p1 = std::get_if<double>(&params[1]);
        const double* p2 = std::get_if<double>(&params[2]);
        if (p0 && p1 && p2)
            return getMatrixU_c(*p0, *p1, *p2);
        return std::nullopt;
    }

    case gCX: return GateMatrix::MatrixCX_c;
    case gCZ: return GateMatrix::MatrixCZ_c;

    case gCP: {
        const double* p0 = std::get_if<double>(&params[0]);
        if (p0)
            return c_matrix_t({
                { 1.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { std::cos(*p0), std::sin(*p0) },
            });
        return std::nullopt;
    }

    default:
        assert(false && "Unsupported constant matrix yet");
        return std::nullopt;
    }
}

inline c_matrix_t getConstantMatrix_up(const up_matrix_t& up) {
    c_matrix_t cmat(up.getSize());
    for (unsigned i = 0; i < up.getSize(); i++) {
        const auto& idx = up.data[i].first;
        const auto& phase = up.data[i].second;
        cmat.data[idx] = { std::cos(phase), std::sin(phase) };
    }
    return cmat;
}

inline std::optional<c_matrix_t> getConstantMatrix_p(
        const p_matrix_t& pmat,
        const std::vector<std::pair<int, double>>& varValues) {
    assert(false && "Not Implemented");
    return std::nullopt;
}
} // anonymous namespace

std::optional<c_matrix_t> GateMatrix::getConstantMatrix(
        const std::vector<std::pair<int, double>>& varValues) const {
    if (const auto* p = std::get_if<gate_params_t>(&_matrix))
        return getConstantMatrix_params(gateKind, *p);
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return getConstantMatrix_up(*p);
    if (const auto* p = std::get_if<c_matrix_t>(&_matrix))
        return *p;
    if (const auto* p = std::get_if<p_matrix_t>(&_matrix))
        return getConstantMatrix_p(*p, varValues);

    assert(false && "getConstantMatrix of an unknown matrix type");
    return std::nullopt;
}

// GateMatrix::getParametrizedMatrix() definition
namespace {

inline p_matrix_t matCvt_gp_to_p(GateKind kind, const gate_params_t& params) {
    switch (kind) {
    case gX: return GateMatrix::MatrixX_p;
    case gY: return GateMatrix::MatrixY_p;
    case gZ: return GateMatrix::MatrixZ_p;
    case gH: return GateMatrix::MatrixH_p;
    case gP: {
        const double* pC = std::get_if<double>(&params[0]);
        const int* pV = std::get_if<int>(&params[0]);
        assert(pC || pV);
        if (pC)
            return p_matrix_t({
                { 1.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { std::cos(*pV), std::sin(*pV) }
            });
        return p_matrix_t({
                { 1.0, 0.0 },
                { 0.0, 0.0 },
                { 0.0, 0.0 },
                { Monomial::Expi(*pV) } // expi(theta)
        });
    }

    case gU: {
        const double* p0C = std::get_if<double>(&params[0]);
        const double* p1C = std::get_if<double>(&params[1]);
        const double* p2C = std::get_if<double>(&params[2]);
        const int* p0V = std::get_if<int>(&params[0]);
        const int* p1V = std::get_if<int>(&params[1]);
        const int* p2V = std::get_if<int>(&params[2]);
        assert(p0C || p0V);
        assert(p1C || p1V);
        assert(p2C || p2V);

        p_matrix_t pmat { {1.0, 0.0}, {-1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0} };
        // theta
        if (p0C) {
            pmat.data[0] *= std::complex<double>(std::cos(*p0C), 0.0);
            pmat.data[1] *= std::complex<double>(std::sin(*p0C), 0.0);
            pmat.data[2] *= std::complex<double>(std::sin(*p0C), 0.0);
            pmat.data[3] *= std::complex<double>(std::cos(*p0C), 0.0);
        } else {
            pmat.data[0] *= Monomial::Cosine(*p0V);
            pmat.data[1] *= Monomial::Sine(*p0V);
            pmat.data[2] *= Monomial::Sine(*p0V);
            pmat.data[3] *= Monomial::Cosine(*p0V);
        }
        // phi
        if (p1C) {
            pmat.data[2] *= std::complex<double>(std::cos(*p1C), std::sin(*p1C));
            pmat.data[3] *= std::complex<double>(std::cos(*p1C), std::sin(*p1C));
        } else {
            pmat.data[2] *= Monomial::Expi(*p1V);
            pmat.data[3] *= Monomial::Expi(*p1V);
        }
        // lambda
        if (p2C) {
            pmat.data[1] *= std::complex<double>(std::cos(*p2C), std::sin(*p2C));
            pmat.data[3] *= std::complex<double>(std::cos(*p2C), std::sin(*p2C));
        } else {
            pmat.data[1] *= Monomial::Expi(*p2V);
            pmat.data[3] *= Monomial::Expi(*p2V);
        }
        return pmat;
    }

    case gCX: return GateMatrix::MatrixCX_p;
    case gCZ: return GateMatrix::MatrixCZ_p;

    default:
        assert(false && "Unsupported constant matrix yet");
        return {};
    }
}

inline p_matrix_t
getParametrizedMatrix_c(const c_matrix_t& cmat) {
    auto edgeSize = cmat.edgeSize();
    p_matrix_t pmat(edgeSize);
    for (size_t i = 0; i < edgeSize * edgeSize; i++)
        pmat.data[i] = Polynomial(cmat.data[i]);
    return pmat;
}

} // anynomous namespace

p_matrix_t GateMatrix::getParametrizedMatrix() const {
    if (const auto *p = std::get_if<gate_params_t>(&_matrix))
        return matCvt_gp_to_p(gateKind, *p);
    if (const auto *p = std::get_if<p_matrix_t>(&_matrix))
        return *p;
    auto cmat = getConstantMatrix();
    assert(cmat.has_value());
    return getParametrizedMatrix_c(cmat.value());
}