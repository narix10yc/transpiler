#ifndef QUENCH_QUANTUM_GATE_H
#define QUENCH_QUANTUM_GATE_H

#include "quench/ComplexMatrix.h"
#include "quench/Polynomial.h"

namespace quench::quantum_gate {

struct matrix_t {
    // parametrised matrix type
    using p_matrix_t = complex_matrix::SquareComplexMatrix<cas::Polynomial>;
    // constant matrix type
    using c_matrix_t = complex_matrix::SquareComplexMatrix<double>;
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
        // std::cerr << "called matrix_t(matrix_t&&)\n";
        
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
        // std::cerr << "called matrix_t(const matrix_t&)\n";

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

    ~matrix_t() {
        destroyMatrix();
    }
};

/// @brief GateMatrix is a wrapper around constant and polynomial-based square
/// matrices. Matrix size will always be a power of 2 so that it represents
/// quantum gates. 
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

    bool checkConsistency() const {
        return (1 << nqubits == N)
            && (matrix.getSize() == N);
    }

    static GateMatrix
    FromName(const std::string& name, const std::vector<double>& params = {});

    bool isConstantMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::C;
    }

    bool isParametrizedMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::P;
    }

    matrix_t::c_matrix_t& cMatrix() {
        assert(isConstantMatrix());
        return matrix.constantMatrix;
    }

    matrix_t::p_matrix_t& pMatrix() {
        assert(isParametrizedMatrix());
        return matrix.parametrizedMatrix;
    }

    int updateNqubits();

    GateMatrix permute(const std::vector<unsigned>& flags) const;
    
    GateMatrix& permuteSelf(const std::vector<unsigned>& flags);

    std::ostream& printMatrix(std::ostream& os) const;
};

class QuantumGate {
public:
    /// The canonical form of qubits is in ascending order
    std::vector<unsigned> qubits;
    GateMatrix matrix;

    QuantumGate() : qubits(), matrix() {}

    QuantumGate(const GateMatrix& matrix, unsigned q)
        : matrix(matrix), qubits({q}) {
        assert(matrix.nqubits == 1);
    }

    QuantumGate(const GateMatrix& matrix, std::initializer_list<unsigned> qubits)
        : matrix(matrix), qubits(qubits) {
        assert(matrix.nqubits == qubits.size());
    }

    QuantumGate(const GateMatrix& matrix, const std::vector<unsigned>& qubits)
        : matrix(matrix), qubits(qubits) {
        assert(matrix.nqubits == qubits.size());
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
        return (matrix.nqubits == qubits.size())
            // && isQubitsSorted()
            && matrix.checkConsistency();
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

};

} // namespace quench::quantum_gate

#endif // QUENCH_QUANTUM_GATE_H