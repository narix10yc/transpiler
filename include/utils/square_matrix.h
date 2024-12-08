#ifndef UTILS_SQUARE_MATRIX_H
#define UTILS_SQUARE_MATRIX_H

#include <memory>
#include <cassert>

namespace utils {

/// @brief square_matrix<T>: a simple wrapper of square matrices. Elements are
/// by default 'not' initialized.
template<typename T>
class square_matrix {
private:
  T* _data;
  size_t _edgeSize; // use size_t here to avoid overflow when accessing elements
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

  T& rc(unsigned r, unsigned c) { return _data[r * _edgeSize + c]; }
  const T& rc(unsigned r, unsigned c) const { return _data[r * _edgeSize + c]; }
};

} // namespace utils

#endif // UTILS_SQUARE_MATRIX_H