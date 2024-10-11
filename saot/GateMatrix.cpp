#include "saot/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <iomanip>
#include <cmath>
#include <type_traits>

using namespace IOColor;
using namespace saot;

GateType String2GateType(const std::string& s) {
    if (s == "x") return gX;
    if (s == "y") return gY;
    if (s == "z") return gZ;
    if (s == "h") return gH;
    if (s == "u3") return gU;
    if (s == "cx") return gCX;
    if (s == "cz") return gCZ;

    assert(false && "Unimplemented String2GateType");
    return gUndef;
}

std::string GateType2String(GateType t) {
    switch (t) {
    case gX: return "x";
    case gY: return "y";
    case gZ: return "z";
    case gH: return "h";
    case gU: return "u3";

    case gCX: return "cx";
    case gCZ: return "cz";
    
    default:
        assert(false && "Unimplemented GateType2String");
        return "undef";
    }
}

// static matrices
const GateMatrix::up_matrix_t GateMatrix::MatrixX_up = GateMatrix::up_matrix_t({
    {1, 0.0}, {0, 0.0}
});

const GateMatrix::up_matrix_t GateMatrix::MatrixY_up = GateMatrix::up_matrix_t({
    {1, -M_PI_2}, {0, M_PI}
});

const GateMatrix::up_matrix_t GateMatrix::MatrixZ_up = GateMatrix::up_matrix_t({
    {0, 0.0}, {1, M_PI}
});

const GateMatrix::up_matrix_t GateMatrix::MatrixCX_up = GateMatrix::up_matrix_t({
    {0, 0.0}, {3, 0.0}, {2, 0.0}, {1, 0.0}
});

const GateMatrix::up_matrix_t GateMatrix::MatrixCZ_up = GateMatrix::up_matrix_t({
    {0, 0.0}, {1, 0.0}, {2, 0.0}, {3, M_PI}
});

const GateMatrix::c_matrix_t GateMatrix::MatrixX_c = GateMatrix::c_matrix_t({
    {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
});

const GateMatrix::c_matrix_t GateMatrix::MatrixY_c({
    {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}
});

const GateMatrix::c_matrix_t GateMatrix::MatrixZ_c = GateMatrix::c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
});

const GateMatrix::c_matrix_t GateMatrix::MatrixH_c = GateMatrix::c_matrix_t({
    {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}
});

const GateMatrix::c_matrix_t GateMatrix::MatrixCX_c = GateMatrix::c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
});

const GateMatrix::c_matrix_t GateMatrix::MatrixCZ_c = GateMatrix::c_matrix_t({
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},
});

int getNumActiveParams(const GateMatrix::params_t& params) {
    for (int i = 0; i < params.size(); i++) {
        if (params[i].index() == 0)
            return i;
    }
    return params.size();
}

GateMatrix GateMatrix::FromName(const std::string& name, const params_t& params) {
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
    if (name == "u3") {
        assert(getNumActiveParams(params) == 3 && "U3 gate has 3 parameters");
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

    assert(false && "Unsupported gate");
    return GateMatrix(gUndef);
}


inline int nqubits_params(GateType ty) {
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
            assert(false && "Unknown GateType nqubits");
            return 0;
    }
}

inline int nqubits_up(const GateMatrix::up_matrix_t& matrix) {
    int nqubits = std::log2(matrix.data.size());
    assert((1 << nqubits) == matrix.data.size());
    return nqubits;
}

inline int nqubits_c(const GateMatrix::c_matrix_t& matrix) {
    int nqubits = std::log2(matrix.getSize());
    assert(1 << nqubits == matrix.getSize());
    return nqubits;
}

inline int nqubits_p(const GateMatrix::p_matrix_t& matrix) {
    int nqubits = std::log2(matrix.getSize());
    assert(1 << nqubits == matrix.getSize());
    return nqubits;
}

int GateMatrix::nqubits() const {
    if (const auto* p = std::get_if<params_t>(&_matrix))
        return nqubits_params(gateTy);
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
    if (const auto* p = std::get_if<params_t>(&_matrix))
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

inline std::ostream&
printMatrix_params(std::ostream& os, const GateMatrix&) {

    return os;
}

inline std::ostream&
printMatrix_up(std::ostream& os, const GateMatrix::up_matrix_t&) {

    return os;
}

inline std::ostream&
printMatrix_c(std::ostream& os, const GateMatrix::c_matrix_t& matrix) {
    const auto& data = matrix.data;
    size_t N = data.size();
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
}

inline std::ostream&
printMatrix_p(std::ostream& os, const GateMatrix::p_matrix_t& matrix) {
    const auto& data = matrix.data;
    size_t N = data.size();
    for (size_t r = 0; r < N; r++) {
        for (size_t c = 0; c < N; c++) {
            os << "[" << r << "," << c << "]: ";
            data[r*N + c].print(os) << "\n";
        }
    }
    return os;
}

std::ostream& GateMatrix::printMatrix(std::ostream& os) const {     
    return std::visit([this, &os](auto&& arg) -> std::ostream& {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, params_t>)
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

std::optional<GateMatrix::up_matrix_t> GateMatrix::getUnitaryPermMatrix() const {
    if (const auto* p = std::get_if<params_t>(&_matrix)) {
        switch (gateTy) {
        case gX: return MatrixX_up;
        case gY: return MatrixY_up;
        case gZ: return MatrixZ_up;
        case gCX: return MatrixCX_up;
        case gCZ: return MatrixCZ_up;

        default:
            return std::nullopt;
        }
    }
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return *p;
    
    return std::nullopt;
}

inline GateMatrix::c_matrix_t getMatrixU_c(double theta, double lambd, double phi) {
    double ctheta = std::cos(theta);
    double stheta = std::sin(theta);
    return GateMatrix::c_matrix_t({
        { ctheta, 0.0 },
        { -std::cos(lambd) * stheta, -std::sin(lambd) * stheta },
        { std::cos(phi) * stheta, std::sin(phi) * stheta },
        { std::cos(phi+lambd) * ctheta, std::sin(phi+lambd) * ctheta }
    });
}

inline std::optional<GateMatrix::c_matrix_t> getConstantMatrix_params(
        GateType ty, const GateMatrix::params_t& params) {
    switch (ty) {
    case gX: return GateMatrix::MatrixX_c;
    case gY: return GateMatrix::MatrixY_c;
    case gZ: return GateMatrix::MatrixZ_c;
    case gH: return GateMatrix::MatrixH_c;

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

    default:
        assert(false && "Unsupported constant matrix yet");
        return std::nullopt;
    }
}

inline GateMatrix::c_matrix_t getConstantMatrix_up(const GateMatrix::up_matrix_t& up) {
    GateMatrix::c_matrix_t cmat(up.getSize());
    for (unsigned i = 0; i < up.getSize(); i++) {
        const auto& idx = up.data[i].first;
        const auto& phase = up.data[i].second;
        cmat.data[idx] = { std::cos(phase), std::sin(phase) };
    }
    return cmat;
}

inline std::optional<GateMatrix::c_matrix_t> getConstantMatrix_p(
        const GateMatrix::p_matrix_t& pmat,
        const std::vector<std::pair<int, double>>& varValues) {
    assert(false && "Not Implemented");
    return std::nullopt;
}

std::optional<GateMatrix::c_matrix_t> GateMatrix::getConstantMatrix(
        const std::vector<std::pair<int, double>>& varValues) const {
    if (const auto* p = std::get_if<params_t>(&_matrix))
        return getConstantMatrix_params(gateTy, *p);
    if (const auto* p = std::get_if<up_matrix_t>(&_matrix))
        return getConstantMatrix_up(*p);
    if (const auto* p = std::get_if<c_matrix_t>(&_matrix))
        return *p;
    if (const auto* p = std::get_if<p_matrix_t>(&_matrix))
        return getConstantMatrix_p(*p, varValues);

    assert(false && "getConstantMatrix of an unknown matrix type");
    return std::nullopt;
}

GateMatrix::p_matrix_t GateMatrix::getParametrizedMatrix() const {
    if (const auto *p = std::get_if<p_matrix_t>(&_matrix))
        return *p;
    assert(0 && "Not Implemented");
    return {};
}