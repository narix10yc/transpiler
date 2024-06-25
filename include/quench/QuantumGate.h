#ifndef QUENCH_QUANTUM_GATE_H
#define QUENCH_QUANTUM_GATE_H

#include "quench/ComplexMatrix.h"
#include "quench/Polynomial.h"

namespace quench::quantum_gate {

class GateMatrix {
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
            std::cerr << "called matrix_t(matrix_t&&)\n";
            
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
            std::cerr << "called matrix_t(const matrix_t&)\n";

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

        matrix_t& operator=(const matrix_t&) = delete;
        matrix_t& operator=(matrix_t&&) = delete;

        ~matrix_t() {
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
        }
    };

public:
    unsigned nqubits;
    size_t N;
    matrix_t matrix;
    GateMatrix() : nqubits(0), N(0), matrix() {}

    GateMatrix(std::initializer_list<complex_matrix::Complex<double>> m);

    static GateMatrix
    FromName(const std::string& name, const std::vector<double>& params);

    bool isConstantMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::C;
    }

    bool isParametrizedMatrix() const {
        return matrix.activeType == matrix_t::ActiveMatrixType::P;
    }

    std::ostream& printMatrix(std::ostream& os) const {
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
};

} // namespace quench::quantum_gate

#endif // QUENCH_QUANTUM_GATE_H