#ifndef SAOT_QUANTUM_GATE_H
#define SAOT_QUANTUM_GATE_H

#include "saot/ComplexMatrix.h"
#include "saot/Polynomial.h"
#include "utils/utils.h"
#include <array>
#include <optional>
#include <variant>
namespace saot {

enum GateType : int {
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

GateType String2GateType(const std::string& s);
std::string GateType2String(GateType t);


/// @brief GateMatrix is a wrapper around constant and polynomial-based square
/// matrices. It does NOT store qubits array, only the number of qubits
/// Consistency requires \p matrix.size() is always a power of 2.
class GateMatrix {
public:
    // specify gate matrix with (up to three) parameters
    using params_t = std::array<std::variant<std::monostate, int, double>, 3>;
    // unitary permutation matrix type
    using up_matrix_t = complex_matrix::UnitaryPermutationMatrix<double>;
    // parametrised matrix type
    using p_matrix_t = complex_matrix::SquareMatrix<saot::Polynomial>;
    // constant matrix type
    using c_matrix_t = complex_matrix::SquareMatrix<std::complex<double>>;
public:
    GateType gateTy;
    std::variant<std::monostate, up_matrix_t, params_t, c_matrix_t, p_matrix_t> _matrix;

    // effectively _matrix.index()
    enum MatrixKind : int {
        MK_NotInitialized = 0,
        MK_ByParameters   = 1,
        MK_UnitaryPerm    = 2,
        MK_Constant       = 3,
        MK_Parametrized   = 4,
    };

    GateMatrix() : gateTy(gUndef), _matrix() {}

    GateMatrix(GateType gateTy, const params_t& params = {})
        : gateTy(gateTy), _matrix(params) {}
        
    GateMatrix(const std::variant<std::monostate, up_matrix_t, params_t, c_matrix_t, p_matrix_t>& m)
        : _matrix(m) { gateTy = GateType(nqubits()); }
    
    static GateMatrix FromName(const std::string& name, const params_t& params = {});

    // bool tryConvertSelfToUnitaryPerm();

    std::optional<up_matrix_t> getUnitaryPermMatrix() const;

    std::optional<c_matrix_t> getConstantMatrix(const std::vector<std::pair<int, double>>& = {}) const;

    p_matrix_t getParametrizedMatrix() const;

    // GateMatrix convertToParametrizedMatrix() const;

    // @brief Get number of qubits
    int nqubits() const;

    /// @brief Approximate matrix elements. Change matrix in-place.
    /// @param level : optimization level. Level 0 turns off everything. Level 1
    /// only applies zero-skipping. Level > 1 also applies to 1 and -1.
    GateMatrix& approximateSelf(int level, double thres = 1e-8);

    GateMatrix permute(const std::vector<int>& flags) const;

    std::ostream& printMatrix(std::ostream& os) const;

    // preset unitary matrices
    static const up_matrix_t MatrixX_up;
    static const up_matrix_t MatrixY_up;
    static const up_matrix_t MatrixZ_up;
    static const up_matrix_t MatrixCX_up;
    static const up_matrix_t MatrixCZ_up;

    // preset constant matrices
    static const c_matrix_t MatrixX_c;
    static const c_matrix_t MatrixY_c;
    static const c_matrix_t MatrixZ_c;
    static const c_matrix_t MatrixH_c;

    static const c_matrix_t MatrixCX_c;
    static const c_matrix_t MatrixCZ_c;
};

class QuantumGate {
private:
    int opCountCache = -1;
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

    /// @brief A.lmatmul(B) will return BA 
    QuantumGate lmatmul(const QuantumGate& other) const;

    int opCount(double zeroSkippingThres = 1e-8);

    bool isConvertibleToUnitaryPermGate() const;
};

} // namespace saot

#endif // SAOT_QUANTUM_GATE_H