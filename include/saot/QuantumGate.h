#ifndef SAOT_QUANTUM_GATE_H
#define SAOT_QUANTUM_GATE_H

#include "saot/ComplexMatrix.h"
#include "saot/Polynomial.h"
#include "utils/utils.h"
#include <array>
#include <optional>
#include <variant>
namespace saot {

enum GateKind : int {
    gUndef = 0,

    // single-qubit gates
    gX  = -10,
    gY  = -11,
    gZ  = -12,
    gH  = -13,
    gP  = -14,
    gRX = -15,
    gRY = -16,
    gRZ = -17,
    gU  = -18,
    
    // two-qubit gates
    gCX   = -100,
    gCNOT = -100,
    gCZ   = -101,
    gSWAP = -102,
    gCP   = -103,

    // general multi-qubit dense gates
    gU1q = 1,
    gU2q = 2,
    gU3q = 3,
    gU4q = 4,
    gU5q = 5,
    gU6q = 6,
    gU7q = 7,
    // to be defined by nqubits directly
};

GateKind String2GateKind(const std::string& s);
std::string GateKind2String(GateKind t);

std::ostream& printConstantMatrix(std::ostream& os,
        const complex_matrix::SquareMatrix<std::complex<double>>& cMat);

std::ostream& printParametrizedMatrix(std::ostream& os,
        const complex_matrix::SquareMatrix<saot::Polynomial>& cMat);


enum ScalarKind : int {
    SK_Zero = 0,
    SK_One = 1,
    SK_MinusOne = -1,
    SK_General = 2,
    SK_ImmValue = 3,
};

class GateMatrix {
public:
    // specify gate matrix with (up to three) parameters
    using gate_params_t = std::array<std::variant<std::monostate, int, double>, 3>;
    // unitary permutation matrix type
    using up_matrix_t = complex_matrix::UnitaryPermutationMatrix<double>;
    // constant matrix type
    using c_matrix_t = complex_matrix::SquareMatrix<std::complex<double>>;
    // parametrised matrix type
    using p_matrix_t = complex_matrix::SquareMatrix<saot::Polynomial>;
    // matrix signature type
    using sig_matrix_t = complex_matrix::SquareMatrix<std::complex<ScalarKind>>;
private:
    enum ConvertibleKind : int {
        Unknown = -1, Convertible = 1, UnConvertible = 0
    };
    struct Cache {
        ConvertibleKind isConvertibleToUpMat;
        up_matrix_t upMat;
        ConvertibleKind isConvertibleToCMat;
        c_matrix_t cMat;
        // gate matrix is always convertible to pMat
        ConvertibleKind isConvertibleToPMat;
        p_matrix_t pMat;

        double sigZeroTol;
        double sigOneTol;
        sig_matrix_t sigMat;

        Cache() : isConvertibleToUpMat(Unknown), upMat(),
                  isConvertibleToCMat(Unknown), cMat(),
                  isConvertibleToPMat(Unknown), pMat(),
                  sigZeroTol(-1.0), sigOneTol(-1.0), sigMat() {}
    };

    mutable Cache cache;

    void computeAndCacheUpMat(double tolerance) const;
    void computeAndCacheCMat() const;
    void computeAndCachePMat() const;
    void computeAndCacheSigMat(double zeroTol, double oneTol) const;
public:
    GateKind gateKind;
    gate_params_t gateParameters;

    GateMatrix() : cache(), gateKind(gUndef), gateParameters() {}

    GateMatrix(GateKind gateKind, const gate_params_t& params = {})
        : cache(), gateKind(gateKind), gateParameters(params) {}
    
    GateMatrix(const up_matrix_t& upMat);
    GateMatrix(const c_matrix_t& upMat);
    GateMatrix(const p_matrix_t& upMat);
    
    static GateMatrix FromName(const std::string& name, const gate_params_t& params = {});

    void permuteSelf(const std::vector<int>& flags);

    // get cached unitary perm matrix object associated with this GateMatrix
    const up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) const {
        if (cache.isConvertibleToUpMat == Unknown)
            computeAndCacheUpMat(tolerance);
        if (cache.isConvertibleToUpMat == UnConvertible)
            return nullptr;
        return &cache.upMat;
    }

    up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) {
        if (cache.isConvertibleToUpMat == Unknown)
            computeAndCacheUpMat(tolerance);
        if (cache.isConvertibleToUpMat == UnConvertible)
            return nullptr;
        return &cache.upMat;
    }

    // get cached constant matrix object associated with this GateMatrix
    const c_matrix_t* getConstantMatrix() const {
        if (cache.isConvertibleToCMat == Unknown)
            computeAndCacheCMat();
        if (cache.isConvertibleToCMat == UnConvertible)
            return nullptr;
        return &cache.cMat;
    }

    c_matrix_t* getConstantMatrix() {
        if (cache.isConvertibleToCMat == Unknown)
            computeAndCacheCMat();
        if (cache.isConvertibleToCMat == UnConvertible)
            return nullptr;
        return &cache.cMat;
    }

    // get cached parametrized matrix object associated with this GateMatrix
    const p_matrix_t& getParametrizedMatrix() const {
        if (cache.isConvertibleToPMat == Unknown)
            computeAndCachePMat();
        assert(cache.isConvertibleToPMat == Convertible);
        return cache.pMat;
    }

    p_matrix_t& getParametrizedMatrix() {
        if (cache.isConvertibleToPMat == Unknown)
            computeAndCachePMat();
        assert(cache.isConvertibleToPMat == Convertible);
        return cache.pMat;
    }

    const sig_matrix_t& getSignatureMatrix(double zeroTol, double oneTol) const {
        if (cache.sigMat.edgeSize() == 0) {
            assert(cache.sigZeroTol < 0.0);
            assert(cache.sigOneTol < 0.0);
            computeAndCacheSigMat(zeroTol, oneTol);
        }
        return cache.sigMat;
    }

    bool isConvertibleToUnitaryPermMatrix(double tolerance) const {
        return getUnitaryPermMatrix(tolerance) != nullptr;
    }

    bool isConvertibleToConstantMatrix() const {
        return getConstantMatrix() != nullptr;
    }

    // @brief Get number of qubits
    int nqubits() const;

    std::ostream& printCMat(std::ostream& os) const {
        const auto* cMat = getConstantMatrix();
        assert(cMat);
        return printConstantMatrix(os, *cMat);
    }

    std::ostream& printPMat(std::ostream& os) const {
        return printParametrizedMatrix(os, getParametrizedMatrix());
    }

    // preset unitary matrices
    static const up_matrix_t MatrixI1_up;
    static const up_matrix_t MatrixI2_up;

    static const up_matrix_t MatrixX_up;
    static const up_matrix_t MatrixY_up;
    static const up_matrix_t MatrixZ_up;
    static const up_matrix_t MatrixCX_up;
    static const up_matrix_t MatrixCZ_up;

    // preset constant matrices
    static const c_matrix_t MatrixI1_c;
    static const c_matrix_t MatrixI2_c;

    static const c_matrix_t MatrixX_c;
    static const c_matrix_t MatrixY_c;
    static const c_matrix_t MatrixZ_c;
    static const c_matrix_t MatrixH_c;

    static const c_matrix_t MatrixCX_c;
    static const c_matrix_t MatrixCZ_c;

    // preset parametrized matrices
    static const p_matrix_t MatrixI1_p;
    static const p_matrix_t MatrixI2_p;

    static const p_matrix_t MatrixX_p;
    static const p_matrix_t MatrixY_p;
    static const p_matrix_t MatrixZ_p;
    static const p_matrix_t MatrixH_p;

    static const p_matrix_t MatrixCX_p;
    static const p_matrix_t MatrixCZ_p;

};

// std::vector<std::complex<ScalarKind>> getScalarKinds(
//         const GateMatrix& gateMatrix, double zeroTol, double oneTol) {
//     std::vector<std::complex<ScalarKind>> skVec;
//     const auto* cMat = gateMatrix.getConstantMatrix();
//     assert(cMat);
//     assert(cMat->edgeSize() > 0);

//     auto edgeSize = cMat->edgeSize();
//     skVec.reserve(edgeSize * edgeSize);

//     for (const auto& cplx : cMat->data) {
//         std::complex<ScalarKind> skCplx(SK_General, SK_General);
//         if (std::abs(cplx.real()) <= zeroTol)
//             skCplx.real(SK_Zero);
//         else if (std::abs(cplx.real() - 1.0) <= oneTol)
//             skCplx.real(SK_One);
//         else if (std::abs(cplx.real() + 1.0) <= oneTol)
//             skCplx.real(SK_MinusOne);

//         if (std::abs(cplx.imag()) <= zeroTol)
//             skCplx.imag(SK_Zero);
//         else if (std::abs(cplx.imag() - 1.0) <= oneTol)
//             skCplx.imag(SK_One);
//         else if (std::abs(cplx.imag() + 1.0) <= oneTol)
//             skCplx.imag(SK_MinusOne);
//         skVec.push_back(skCplx);
//     }

//     return skVec;
// }

class QuantumGate {
private:
    mutable int opCountCache = -1;
public:
    /// The canonical form of qubits is in ascending order
    std::vector<int> qubits;
    GateMatrix gateMatrix;

    QuantumGate() : qubits(), gateMatrix() {}

    QuantumGate(const GateMatrix& gateMatrix, int q)
        : gateMatrix(gateMatrix), qubits({q}) {
        assert(gateMatrix.nqubits() == 1);
    }

    QuantumGate(const GateMatrix& gateMatrix, std::initializer_list<int> qubits)
        : gateMatrix(gateMatrix), qubits(qubits) {
        assert(gateMatrix.nqubits() == qubits.size());
        sortQubits();
    }

    QuantumGate(const GateMatrix& gateMatrix, const std::vector<int>& qubits)
        : gateMatrix(gateMatrix), qubits(qubits) {
        assert(gateMatrix.nqubits() == qubits.size());
        sortQubits();
    }

    bool isQubitsSorted() const {
        if (qubits.empty())
            return true;
        for (unsigned i = 0; i < qubits.size()-1; i++) {
            if (qubits[i+1] <= qubits[i])
                return false;
        }
        return true;
    }

    bool checkConsistency() const {
        return (gateMatrix.nqubits() == qubits.size());
    }

    std::ostream& displayInfo(std::ostream& os) const;

    int findQubit(int q) const {
        for (unsigned i = 0; i < qubits.size(); i++) {
            if (qubits[i] == q)
                return i;
        }
        return -1;
    }

    void sortQubits();

    /// @brief B.lmatmul(A) will return AB
    QuantumGate lmatmul(const QuantumGate& other) const;

    int opCount(double zeroSkippingThres = 1e-8) const;

    bool isConvertibleToUnitaryPermGate(double tolerance) const {
        return gateMatrix.isConvertibleToUnitaryPermMatrix(tolerance);
    }

    bool isConvertibleToConstantGate() const {
        return gateMatrix.isConvertibleToConstantMatrix();
    }
};

} // namespace saot

#endif // SAOT_QUANTUM_GATE_H