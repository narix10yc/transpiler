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

std::ostream& saot::printConstantMatrix(std::ostream& os, const c_matrix_t& cMat) {
    const auto edgeSize = cMat.edgeSize();
    const auto& data = cMat.data;
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

std::ostream& saot::printParametrizedMatrix(std::ostream& os, const p_matrix_t& pMat) {
    auto edgeSize = pMat.edgeSize();
    for (size_t r = 0; r < edgeSize; r++) {
        for (size_t c = 0; c < edgeSize; c++) {
            os << "[" << r << "," << c << "]: ";
            pMat.data[r*edgeSize + c].print(os) << "\n";
        }
    }
    return os;
}

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
    unsigned s = params.size();
    for (unsigned i = 0; i < s; i++) {
        if (params[i].index() == 0)
            return i;
    }
    return s;
}

GateMatrix::GateMatrix(const up_matrix_t& upMat) : cache(), gateParameters() {
    auto size = upMat.getSize();
    gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
    assert(1 << static_cast<int>(gateKind) == size);

    cache.upMat = upMat;
    cache.isConvertibleToUpMat = Convertible;
}

GateMatrix::GateMatrix(const c_matrix_t& cMat) : cache(), gateParameters() {
    auto size = cMat.edgeSize();
    gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
    assert(1 << static_cast<int>(gateKind) == size);

    cache.cMat = cMat;
    cache.isConvertibleToCMat = Convertible;
}

GateMatrix::GateMatrix(const p_matrix_t& pMat) : cache(), gateParameters() {
    auto size = pMat.edgeSize();
    gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
    assert(1 << static_cast<int>(gateKind) == size);

    cache.pMat = pMat;
    cache.isConvertibleToPMat = Convertible;
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

void GateMatrix::permuteSelf(const std::vector<int>& flags) {
    switch (gateKind) {
    case gX: assert(flags.size() == 1); return;
    case gY: assert(flags.size() == 1); return;
    case gZ: assert(flags.size() == 1); return;
    case gH: assert(flags.size() == 1); return;
    case gP: assert(flags.size() == 1); return;
    case gU: assert(flags.size() == 1); return;

    case gCX: assert(flags.size() == 2); return;
    case gCZ: assert(flags.size() == 2); return;
    case gSWAP: assert(flags.size() == 2); return;
    case gCP: assert(flags.size() == 2); return;
    
    default:
        break;
    }

    assert(gateKind >= 1);
    if (cache.isConvertibleToUpMat == Convertible)
        cache.upMat = cache.upMat.permute(flags);
    if (cache.isConvertibleToCMat == Convertible)
        cache.cMat = cache.cMat.permute(flags);
    if (cache.isConvertibleToPMat == Convertible)
        cache.pMat = cache.pMat.permute(flags);
}

int GateMatrix::nqubits() const {
    switch (gateKind) {
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
            assert(gateKind >= 1);
            return static_cast<int>(gateKind);
    }
}

namespace { // matrix conversion

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
        assert(false && "Unsupported cvtMat_gp_to_p yet");
        return {};
    }
}

inline c_matrix_t matCvt_up_to_c(const up_matrix_t& up) {
    c_matrix_t cmat(up.getSize());
    for (unsigned i = 0; i < up.getSize(); i++) {
        const auto& idx = up.data[i].first;
        const auto& phase = up.data[i].second;
        cmat.data[idx] = { std::cos(phase), std::sin(phase) };
    }
    return cmat;
}

} // anynomous namespace

// Two paths lead to upMat: gpMat or cMat
void GateMatrix::computeAndCacheUpMat() const {
    assert(cache.isConvertibleToCMat == Unknown);

    switch (gateKind) {
    case gX:  {
        cache.upMat = MatrixX_up;
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gY: {
        cache.upMat = MatrixY_up;
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gZ: {
        cache.upMat = MatrixZ_up;
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gP: {
        const double* plambd = std::get_if<double>(&gateParameters[0]);
        if (plambd == nullptr) {
            cache.isConvertibleToUpMat = UnConvertible;
            return;
        }
        cache.upMat = up_matrix_t {{0, 0.0}, {1, *plambd}};
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gCX: {
        cache.upMat = MatrixCX_up;
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gCZ: {
        cache.upMat = MatrixCZ_up;
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    case gCP: {
        const double* plambd = std::get_if<double>(&gateParameters[0]);
        if (plambd == nullptr) {
            cache.isConvertibleToUpMat = UnConvertible;
            return;
        }
        cache.upMat = up_matrix_t {{0, 0.0}, {0, 0.0}, {0, 0.0}, {1, *plambd}};
        cache.isConvertibleToUpMat = Convertible;
        return;
    }
    default:
        break;
    }
    
    const auto* cMat = getConstantMatrix();
    if (cMat == nullptr) {
        cache.isConvertibleToUpMat = UnConvertible;
        return;
    }
    const auto edgeSize = cMat->edgeSize();
    cache.upMat = up_matrix_t(edgeSize);
    for (size_t r = 0; r < edgeSize; r++) {
        bool rowFlag = false;
        for (size_t c = 0; c < edgeSize; c++) {
            const auto& cplx = cMat->data[r * edgeSize + c];
            if (cplx != std::complex<double>{ 0.0, 0.0 }) {
                if (rowFlag) {
                    cache.isConvertibleToUpMat = UnConvertible;
                    return;
                }
                rowFlag = true;
                cache.upMat.data[r] = { c, std::atan2(cplx.imag(), cplx.real()) };
            }
        }
        if (!rowFlag) {
            cache.isConvertibleToUpMat = UnConvertible;
            return;
        }
    }
    cache.isConvertibleToUpMat = Convertible;
    return;
}

void GateMatrix::computeAndCacheCMat() const {
    assert(cache.isConvertibleToUpMat == Unknown);
    // try convert from upMat
    if (cache.isConvertibleToUpMat == Convertible) {
        const auto edgeSize = cache.upMat.getSize();
        cache.cMat = c_matrix_t(edgeSize);
        for (unsigned i = 0; i < edgeSize; ++i) {
            const auto& idx = cache.upMat.data[i].first;
            const auto& phase = cache.upMat.data[i].second;
            cache.cMat.data[idx] = { std::cos(phase), std::sin(phase) };
        }
        cache.isConvertibleToCMat = Convertible;
        return;
    }

    // try convert from gpMat
    switch (gateKind) {
    case gX: {
        cache.cMat = MatrixX_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gY: {
        cache.cMat = MatrixY_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gZ: {
        cache.cMat = MatrixZ_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gH: {
        cache.cMat = MatrixH_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gP: {
        const double* p = std::get_if<double>(&gateParameters[0]);
        if (p == nullptr) {
            cache.isConvertibleToCMat = UnConvertible;
            return;
        }
        cache.cMat = c_matrix_t({
                { 1.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { std::cos(*p), std::sin(*p) }
            });
        cache.isConvertibleToCMat = Convertible;
        return;
    }

    case gU: {
        const double* p0 = std::get_if<double>(&gateParameters[0]);
        const double* p1 = std::get_if<double>(&gateParameters[1]);
        const double* p2 = std::get_if<double>(&gateParameters[2]);
        if (p0 && p1 && p2) {
            cache.cMat = c_matrix_t({
                { std::cos(*p0), 0.0 },
                { -std::cos(*p2) * std::sin(*p0), -std::sin(*p2) * std::sin(*p0) },
                { std::cos(*p1) * std::sin(*p0), std::sin(*p1) * std::sin(*p0) },
                { std::cos(*p1 + *p2) * std::cos(*p0), std::sin(*p1 + *p2) * std::cos(*p0) }
            });
            cache.isConvertibleToCMat = Convertible;
            return;
        }
        cache.isConvertibleToCMat = UnConvertible;
        return;
    }
    case gCX: {
        cache.cMat = MatrixCX_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gCZ: {
        cache.cMat = MatrixCZ_c;
        cache.isConvertibleToCMat = Convertible;
        return;
    }
    case gCP: {
        const double* p0 = std::get_if<double>(&gateParameters[0]);
        if (p0 == nullptr) {
            cache.isConvertibleToCMat = UnConvertible;
            return;
        }
        cache.cMat = c_matrix_t({
                { 1.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 },
                { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { std::cos(*p0), std::sin(*p0) },
            });
        cache.isConvertibleToCMat = Convertible;
        return;
    }

    default:
        assert(false && "Unsupported constant matrix yet");
        cache.isConvertibleToCMat = UnConvertible;
        return;
    }
}

void GateMatrix::computeAndCachePMat() const {
    if (cache.isConvertibleToCMat == Convertible) {
        auto edgeSize = cache.cMat.edgeSize();
        cache.pMat = p_matrix_t(edgeSize);
        for (size_t i = 0; i < edgeSize * edgeSize; ++i)
            cache.pMat.data[i] = Polynomial(cache.cMat.data[i]);
        cache.isConvertibleToPMat = Convertible;
        return;
    }
    cache.pMat = matCvt_gp_to_p(gateKind, gateParameters);
    cache.isConvertibleToPMat = Convertible;
    return;
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

}
