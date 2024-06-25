#ifndef QUENCH_COMPLEX_MATRIX_H
#define QUENCH_COMPLEX_MATRIX_H

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace quench::complex_matrix {

template<typename real_t>
class Complex {
public:
    real_t real, imag;
    Complex() : real(), imag() {}
    Complex(real_t real, real_t imag) : real(real), imag(imag) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imzg);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    Complex& operator*=(const Complex& other) {
        real_t r = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = r;
        return *this;
    }
};

template<typename real_t>
class SquareComplexMatrix {
    size_t size;
public:
    using complex_t = Complex<real_t>;
    std::vector<Complex<real_t>> data;

    SquareComplexMatrix() : size(0), data() {}
    SquareComplexMatrix(size_t size) : size(size), data(size * size) {}
    SquareComplexMatrix(std::initializer_list<complex_t> data)
        : size(std::sqrt(data.size())), data(data) {
        assert(size * size == data.size() && "data.size() should be a perfect square");
    }

    size_t getSize() const { return size; }
    
    bool checkSizeMatch() const {
        return data.size() == size * size;
    }

    static SquareComplexMatrix Identity(size_t size) {
        SquareComplexMatrix m;
        for (size_t r = 0; r < size; r++)
            m.data[r*size + r].real = 1;
        return m;
    }

    SquareComplexMatrix matmul(const SquareComplexMatrix& other) {
        assert(size == other.size);

        SquareComplexMatrix m(size);
        for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
        for (size_t k = 0; k < size; k++) {
            // C_{ij} = A_{ik} B_{kj}
            m.data[i*size + j] += data[i*size + k] * other.data[k*size + j];
        } } }
        return m;
    }

    SquareComplexMatrix kron(const SquareComplexMatrix& other) const {
        size_t lsize = size;
        size_t rsize = other.size;
        size_t msize = lsize * rsize;
        SquareComplexMatrix m(msize);
        for (size_t lr = 0; lr < lsize; lr++) {
        for (size_t lc = 0; lc < lsize; lc++) {
        for (size_t rr = 0; rr < rsize; rr++) {
        for (size_t rc = 0; rc < rsize; rc++) {
            size_t r = lr * rsize + rr;
            size_t c = lc * rsize + rc;
            m.data[r*msize + c] = data[lr*lsize + lc] * other.data[rr*rsize + rc];
        } } } }
        return m;
    }

    SquareComplexMatrix leftKronI() const {
        SquareComplexMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(i*size + r) * size * size + (i*size + c)] = data[r*size + c];
        } } }
        return m;
    }

    SquareComplexMatrix rightKronI() const {
        SquareComplexMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(r*size + i) * size * size + (c*size + i)] = data[r*size + c];
        } } }
        return m;
    }

    SquareComplexMatrix swapTargetQubits() const {
        assert(size == 4);
        return {{data[ 0], data[ 2], data[ 1], data[ 3],
                 data[ 8], data[10], data[ 9], data[11],
                 data[ 4], data[ 6], data[ 5], data[ 7],
                 data[12], data[14], data[13], data[15]}};
    }
};

// class GateMatrix {
// public:
//     unsigned nqubits;
//     size_t N;
//     std::vector<Complex<Polynomial>> matrix;

//     GateMatrix() : nqubits(0), N(1), matrix(1) {}
//     GateMatrix(unsigned nqubits)
//         : nqubits(nqubits), N(1 << nqubits), matrix(1 << (nqubits*2)) {}
//     GateMatrix(std::initializer_list<Complex<Polynomial>> matrix)
//         : matrix(matrix) { int r = updateNQubits(); assert(r > 0); }

//     /// @brief update nqubits and N based on matrix.
//     /// @return if matrix represents a (2**n) * (2**n) matrix, then return n
//     /// (number of qubits). Otherwise, return -1 if matrix is empty; -2 if
//     /// matrix.size() is not a perfect square; -3 if matrix represents an N * N
//     /// matrix, but N is not a power of two.
//     int updateNQubits() {
//         if (matrix.empty())
//             return -1;
//         size_t size = static_cast<size_t>(std::sqrt(matrix.size()));
//         if (size * size == matrix.size()) {
//             N = size;
//             if ((N & (N-1)) == 0) {
//                 nqubits = static_cast<unsigned>(std::log2(N));
//                 return nqubits;
//             }
//             return -3;
//         }
//         return -2;
//     }
    
//     bool checkSizeMatch() const {
//         return (N == (1 << nqubits) && matrix.size() == N * N);
//     }

//     static GateMatrix Identity(unsigned nqubits) {
//         GateMatrix m(nqubits);
//         for (size_t r = 0; r < m.N; r++)
//             m.matrix[r*m.N + r].real = { 1.0 };
//         return m;
//     }

//     GateMatrix matmul(const GateMatrix& other) const {
//         assert(checkSizeMatch());
//         assert(other.checkSizeMatch());
//         assert(nqubits == other.nqubits && N == other.N);

//         GateMatrix m(nqubits);
//         for (size_t i = 0; i < N; i++) {
//         for (size_t j = 0; j < N; j++) {
//         for (size_t k = 0; k < N; k++) {
//             // C_{ij} = A_{ik} B_{kj}
//             m.matrix[i*N + j] += matrix[i*N + k] * other.matrix[k*N + j];
//         } } }
//         return m;
//     }

//     GateMatrix kron(const GateMatrix& other) const {
//         assert(checkSizeMatch());
//         assert(other.checkSizeMatch());

//         size_t lsize = N;
//         size_t rsize = other.N;
//         size_t msize = lsize * rsize;
//         GateMatrix m(nqubits + other.nqubits);
//         for (size_t lr = 0; lr < lsize; lr++) {
//         for (size_t lc = 0; lc < lsize; lc++) {
//         for (size_t rr = 0; rr < rsize; rr++) {
//         for (size_t rc = 0; rc < rsize; rc++) {
//             size_t r = lr * rsize + rr;
//             size_t c = lc * rsize + rc;
//             m.matrix[r*msize + c] = matrix[lr*lsize + lc] * other.matrix[rr*rsize + rc];
//         } } } }
//         return m;
//     }

//     GateMatrix leftKronI() const {
//         assert(checkSizeMatch());

//         GateMatrix m(2 * nqubits);
//         for (size_t i = 0; i < N; i++) {
//         for (size_t r = 0; r < N; r++) {
//         for (size_t c = 0; c < N; c++) {
//             m.matrix[(i*N + r) * N * N + (i*N + c)] = matrix[r*N + c];
//         } } }
//         return m;
//     }

//     GateMatrix rightKronI() const {
//         assert(checkSizeMatch());

//         GateMatrix m(2 * nqubits);
//         for (size_t i = 0; i < N; i++) {
//         for (size_t r = 0; r < N; r++) {
//         for (size_t c = 0; c < N; c++) {
//             m.matrix[(r*N + i) * N * N + (c*N + i)] = matrix[r*N + c];
//         } } }
//         return m;
//     }

//     GateMatrix swapTargetQubits() const {
//         assert(nqubits == 2 && N == 4);

//         return {{matrix[ 0], matrix[ 2], matrix[ 1], matrix[ 3],
//                  matrix[ 8], matrix[10], matrix[ 9], matrix[11],
//                  matrix[ 4], matrix[ 6], matrix[ 5], matrix[ 7],
//                  matrix[12], matrix[14], matrix[13], matrix[15]}};
//     }

//     std::ostream& print(std::ostream& os) const {
//         for (size_t r = 0; r < N; r++) {
//             for (size_t c = 0; c < N; c++) {
//                 const auto& re = matrix[r*N + c].real;
//                 const auto& im = matrix[r*N + c].imag;
//                 re.print(os) << " + ";
//                 im.print(os) << "i ";
//             }
//             os << "\n";
//         }
//         return os;
//     }

//     static GateMatrix
//     FromName(const std::string& name, const std::vector<double>& params);
// };



} // namespace quench::complex_matrix

#endif // QUENCH_COMPLEX_MATRIX_H