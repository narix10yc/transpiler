#ifndef UTILS_SQUARE_MATRIX_H
#define UTILS_SQUARE_MATRIX_H

#include <memory>
#include <cassert>
#include <complex>
#include <iostream>

namespace utils {

/// @brief square_matrix<T>: a simple wrapper of square matrices. Elements are
/// by default 'not' initialized.
template<typename T>
class square_matrix {
private:
  T* _data;
  size_t _edgeSize; // use size_t here to avoid overflow
public:
  square_matrix(unsigned edgeSize)
      : _data((T*)std::malloc((size_t)(edgeSize) * edgeSize * sizeof(T)))
      , _edgeSize(edgeSize) {
    assert(_data != nullptr);
  }

  ~square_matrix() { std::free(_data); }

  square_matrix(const square_matrix&) { assert(0 && "Not Implemented"); }
  square_matrix(square_matrix&&) { assert(0 && "Not Implemented"); }
  
  square_matrix& operator=(const square_matrix&) { assert(0 && "Not Implemented"); }
  square_matrix& operator=(square_matrix&&) { assert(0 && "Not Implemented"); }

  T* data() { return _data; }
  const T* data() const { return _data; }

  T* row(unsigned r) {
    assert(r < _edgeSize);
    return _data + r * _edgeSize;
  }
  const T* row(unsigned r) const {
    assert(r < _edgeSize);
    return _data + r * _edgeSize;
  }

  // get element specified by row and column
  T& rc(unsigned r, unsigned c) {
    assert(r < _edgeSize && c < _edgeSize);
    return _data[r * _edgeSize + c];
  }

  // get element specified by row and column
  const T& rc(unsigned r, unsigned c) const {
    assert(r < _edgeSize && c < _edgeSize);
    return _data[r * _edgeSize + c];
  }

  // get element specified by row and column
  T& operator()(unsigned r, unsigned c) { return rc(r, c); }
  
  // get element specified by row and column
  const T& operator()(unsigned r, unsigned c) const { return rc(r, c); }

  size_t edgeSize() const { return _edgeSize; }

  size_t sizeInBytes() const { return _edgeSize * _edgeSize * sizeof(T); }
};

square_matrix<std::complex<double>> randomUnitaryMatrix(unsigned edgeSize);

std::ostream& printComplexMatrixF64(
    std::ostream& os,
    const square_matrix<std::complex<double>>& matrix);

inline std::ostream& printComplexMatrixF64(
    const square_matrix<std::complex<double>>& matrix) {
  return printComplexMatrixF64(std::cerr, matrix);
}

} // namespace utils

#endif // UTILS_SQUARE_MATRIX_H