#ifndef QUENCH_QUANTUM_GATE_H
#define QUENCH_QUANTUM_GATE_H

#include "quench/ComplexMatrix.h"
#include "quench/Polynomial.h"
#include "utils/utils.h"

namespace quench::quantum_gate {

struct matrix_t {
    // parametrised matrix type
    using p_matrix_t = complex_matrix::SquareMatrix<cas::Polynomial>;
    // constant matrix type
    using c_matrix_t = complex_matrix::SquareMatrix<std::complex<double>>;
    union {
        p_matrix_t parametrizedMatrix;
        c_matrix_t constantMatrix;
    };
    enum class ActiveMatrixType { P, C, None } activeType;

    matrix_t() : activeType(ActiveMatrixType::None) {}

    size_t getSize() const {
        switch (activeType) {
        case ActiveMatrixType::P: return parametrizedMatrix.getSize();
        case ActiveMatrixType::C: return constantMatrix.getSize();
        default: return 0;
        }
    }

    matrix_t(matrix_t&& other) : activeType(other.activeType) {
        switch (other.activeType) {
        case ActiveMatrixType::P:
            new (&parametrizedMatrix) p_matrix_t(std::move(other.parametrizedMatrix));
            break;
        case ActiveMatrixType::C:
            new (&constantMatrix) c_matrix_t(std::move(other.constantMatrix));
            break;
        default:
            break;
        }
    }

    matrix_t(const matrix_t& other) : activeType(other.activeType) {
        switch (other.activeType) {
        case ActiveMatrixType::P:
            new (&parametrizedMatrix) p_matrix_t(other.parametrizedMatrix);
            break;
        case ActiveMatrixType::C:
            new (&constantMatrix) c_matrix_t(other.constantMatrix);
            break;
        default:
            break;
        }
    }

    void destroyMatrix(ActiveMatrixType newActiveType = ActiveMatrixType::None) {
        switch (activeType) {
        case ActiveMatrixType::P:
            parametrizedMatrix.~p_matrix_t();
            break;
        case ActiveMatrixType::C:
            constantMatrix.~c_matrix_t();
            break;
        default:
            break;
        }
        activeType = newActiveType;
    }

    matrix_t& operator=(const matrix_t& other) {
        if (this == &other)
            return *this;
        destroyMatrix(other.activeType);
        switch (other.activeType) {
        case ActiveMatrixType::P:
            new (&parametrizedMatrix) p_matrix_t(other.parametrizedMatrix);
            break;
        case ActiveMatrixType::C:
            new (&constantMatrix) c_matrix_t(other.constantMatrix);
            break;
        default:
            break;
        }
        return *this;
    }

    matrix_t& operator=(matrix_t&& other) {
        if (this == &other)
            return *this;
        destroyMatrix(other.activeType);
        switch (other.activeType) {
        case ActiveMatrixType::P:
            new (&parametrizedMatrix) p_matrix_t(std::move(other.parametrizedMatrix));
            break;
        case ActiveMatrixType::C:
            new (&constantMatrix) c_matrix_t(std::move(other.constantMatrix));
            break;
        default:
            break;
        }
        return *this;
    }

    matrix_t& operator=(const c_matrix_t& cMatrix) {
        if (activeType == ActiveMatrixType::P)
            parametrizedMatrix.~p_matrix_t();
        activeType = ActiveMatrixType::C;
        new (&constantMatrix) c_matrix_t(cMatrix);
        return *this;
    }

    matrix_t& operator=(c_matrix_t&& cMatrix) {
        if (activeType == ActiveMatrixType::P)
            parametrizedMatrix.~p_matrix_t();
        activeType = ActiveMatrixType::C;
        new (&constantMatrix) c_matrix_t(std::move(cMatrix));
        return *this;
    }

    matrix_t& operator=(const p_matrix_t& pMatrix) {
        if (activeType == ActiveMatrixType::C)
            constantMatrix.~c_matrix_t();
        activeType = ActiveMatrixType::P;
        new (&parametrizedMatrix) p_matrix_t(pMatrix);
        return *this;
    }

    matrix_t& operator=(p_matrix_t&& pMatrix) {
        if (activeType == ActiveMatrixType::C)
            constantMatrix.~c_matrix_t();
        activeType = ActiveMatrixType::P;
        new (&parametrizedMatrix) p_matrix_t(std::move(pMatrix));
        return *this;
    }

    ~matrix_t() {
        destroyMatrix();
    }
};

class GateParameter {
public:
    std::string variableName;
    std::complex<double> constant;
    bool isConstant;

    GateParameter(const std::string& variableName)
        : variableName(variableName), isConstant(false) {}
    GateParameter(std::complex<double> constant)
        : constant(constant), isConstant(true) {}

    std::ostream& print(std::ostream& os) const {
        if (isConstant)
            return utils::print_complex(os, constant, 12);
        return os << variableName;
    }
};

/// @brief GateMatrix is a wrapper around constant and polynomial-based square
/// matrices. It does NOT store qubits array, only the number of qubits
/// Consistency requires matrix size is always a power of 2.
class GateMatrix {
public:
    unsigned nqubits;
    size_t N;
    matrix_t matrix;
    GateMatrix() : nqubits(0), N(0), matrix() {}

    GateMatrix(const matrix_t::c_matrix_t& cMatrix) {
        matrix = cMatrix;
        updateNqubits();
    }

    GateMatrix(matrix_t::c_matrix_t&& cMatrix) {
        matrix = std::move(cMatrix);
        updateNqubits();
    }

    GateMatrix(const matrix_t::p_matrix_t& pMatrix) {
        matrix = pMatrix;
        updateNqubits();
    }

    GateMatrix(matrix_t::p_matrix_t&& pMatrix) {
        matrix = std::move(pMatrix);
        updateNqubits();
    }

    bool checkConsistency() const {
        return (1 << nqubits == N)
            && (matrix.getSize() == N);
    }

    static GateMatrix
    FromName(const std::string& name, const std::vector<double>& params = {});

    inline bool isConstantMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::C;
    }

    inline bool isParametrizedMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::P;
    }

    void convertToParametrizedMatrix() {
        if (isParametrizedMatrix())
            return;
        assert(isConstantMatrix());

        size_t size = matrix.constantMatrix.getSize();
        matrix_t::p_matrix_t pmat(size);
        for (size_t i = 0; i < size*size; i++)
            pmat.data[i] = cas::Polynomial(matrix.constantMatrix.data[i]);
        
        matrix = std::move(pmat);
    }

    int updateNqubits();

    /// @brief Approximate matrix elements. Change matrix in-place.
    /// @param level : optimization level. Level 0 turns off everything. Level 1
    /// only applies zero-skipping. Level > 1 also applies to 1 and -1.
    GateMatrix& approximateSelf(int level, double thres = 1e-8);

    GateMatrix permute(const std::vector<unsigned>& flags) const;

    std::ostream& printMatrix(std::ostream& os) const;
};

class QuantumGate {
private:
    int opCountCache = -1;
public:
    /// The canonical form of qubits is in ascending order
    std::vector<unsigned> qubits;
    GateMatrix gateMatrix;

    QuantumGate() : qubits(), gateMatrix() {}

    QuantumGate(const GateMatrix& gateMatrix, unsigned q)
        : gateMatrix(gateMatrix), qubits({q}) {
        assert(gateMatrix.nqubits == 1);
    }

    QuantumGate(const GateMatrix& gateMatrix, std::initializer_list<unsigned> qubits)
        : gateMatrix(gateMatrix), qubits(qubits) {
        assert(gateMatrix.nqubits == qubits.size());
        sortQubits();
    }

    QuantumGate(const GateMatrix& gateMatrix, const std::vector<unsigned>& qubits)
        : gateMatrix(gateMatrix), qubits(qubits) {
        assert(gateMatrix.nqubits == qubits.size());
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
        return (gateMatrix.nqubits == qubits.size())
            // && isQubitsSorted()
            && gateMatrix.checkConsistency();
    }

    std::ostream& displayInfo(std::ostream& os) const;

    int findQubit(unsigned q) const {
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

    matrix_t::c_matrix_t& getCMatrix() {
        if (!gateMatrix.isConstantMatrix())
            throw "calling getCMatrix for a not-constant matrix";
        return gateMatrix.matrix.constantMatrix;
    }

};

} // namespace quench::quantum_gate

#endif // QUENCH_QUANTUM_GATE_H