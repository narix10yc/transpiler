#ifndef QUENCH_COMPLEX_MATRIX_H
#define QUENCH_COMPLEX_MATRIX_H

#include <vector>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>

namespace quench::complex_matrix {

/// @brief SquareMatrix class is a wrapper of data : vector<data_t> with a 
/// private member called 'size'. data.size() should always be a perfect square,
/// equaling to size * size. If manually update data by, for example,
/// data.push_back, it is required to call updateSize() which will check
/// dimension consistency.
/// @tparam data_t 
template<typename data_t>
class SquareMatrix {
    size_t size;
public:
    std::vector<data_t> data;

    SquareMatrix() : size(0), data() {}
    SquareMatrix(size_t size) : size(size), data(size * size) {}
    SquareMatrix(std::initializer_list<data_t> data)
            : size(std::sqrt(data.size())), data(data) {
        assert(size * size == data.size()
               && "data.size() should be a perfect square");
    }

    void updateSize() {
        if (size * size != data.size()) {
            size = std::sqrt(data.size());
            assert(size * size == data.size());
        }
    }
    
    size_t getSize() const { return size; }
    
    bool checkSizeMatch() const {
        return data.size() == size * size;
    }

    static SquareMatrix Identity(size_t size) {
        SquareMatrix m;
        for (size_t r = 0; r < size; r++)
            m.data[r*size + r].real = 1;
        return m;
    }

    SquareMatrix matmul(const SquareMatrix& other) {
        assert(size == other.size);

        SquareMatrix m(size);
        for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
        for (size_t k = 0; k < size; k++) {
            // C_{ij} = A_{ik} B_{kj}
            m.data[i*size + j] += data[i*size + k] * other.data[k*size + j];
        } } }
        return m;
    }

    SquareMatrix kron(const SquareMatrix& other) const {
        size_t lsize = size;
        size_t rsize = other.size;
        size_t msize = lsize * rsize;
        SquareMatrix m(msize);
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

    SquareMatrix leftKronI() const {
        SquareMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(i*size + r) * size * size + (i*size + c)] = data[r*size + c];
        } } }
        return m;
    }

    SquareMatrix rightKronI() const {
        SquareMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(r*size + i) * size * size + (c*size + i)] = data[r*size + c];
        } } }
        return m;
    }

    SquareMatrix swapTargetQubits() const {
        assert(size == 4);
        return {{data[ 0], data[ 2], data[ 1], data[ 3],
                 data[ 8], data[10], data[ 9], data[11],
                 data[ 4], data[ 6], data[ 5], data[ 7],
                 data[12], data[14], data[13], data[15]}};
    }
};


} // namespace quench::complex_matrix

#endif // QUENCH_COMPLEX_MATRIX_H