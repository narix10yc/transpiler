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


} // namespace quench::complex_matrix

#endif // QUENCH_COMPLEX_MATRIX_H