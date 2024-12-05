#ifndef SAOT_COMPLEX_MATRIX_H
#define SAOT_COMPLEX_MATRIX_H

#include "utils/utils.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace saot::complex_matrix {

/// @brief A square matrix that each row or column has exactly one non-zero
/// entry of the form expi(phi).
/// @param data The (i, data[i].first) entry of the matrix is
/// expi(data[i].second)
template <typename data_t> class UnitaryPermutationMatrix {
public:
  std::vector<std::pair<size_t, data_t>> data;

  UnitaryPermutationMatrix() : data() {}
  UnitaryPermutationMatrix(size_t size) : data(size) {}
  UnitaryPermutationMatrix(const std::vector<std::pair<size_t, data_t>>& data)
      : data(data) {}
  UnitaryPermutationMatrix(
      std::initializer_list<std::pair<size_t, data_t>> data)
      : data(data) {}

  size_t getSize() const { return data.size(); }

  static UnitaryPermutationMatrix Identity(size_t size) {
    UnitaryPermutationMatrix m(size);
    for (size_t i = 0; i < size; i++)
      m.data[i] = std::make_pair<size_t, data_t>(i, 0.0);
    return m;
  }

  UnitaryPermutationMatrix permute(const std::vector<int>& flags) const {
    if (std::all_of(flags.begin(), flags.end(),
                    [&flags](int i) { return flags[i] == i; }))
      return UnitaryPermutationMatrix(*this);

    const auto permuteIndex = [&flags, k = flags.size()](size_t idx) -> size_t {
      size_t newIdx = 0;
      for (unsigned b = 0; b < k; b++) {
        newIdx += ((idx & (1ULL << b)) >> b) << flags[b];
      }
      return newIdx;
    };

    UnitaryPermutationMatrix matrix(data.size());
    for (size_t r = 0; r < data.size(); r++)
      matrix.data[permuteIndex(r)] = {permuteIndex(data[r].first),
                                      data[r].second};

    return matrix;
  }
};

/// @brief SquareMatrix class is a wrapper of data : vector<data_t> with a
/// private member called 'size'. data.size() should always be a perfect square,
/// equaling to edgeSize * edgeSize. If manually update data by, for example,
/// data.push_back, it is required to call updateSize() which will check
/// dimension consistency.
/// @tparam data_t
template <typename data_t> class SquareMatrix {
  size_t _edgeSize;

public:
  std::vector<data_t> data;

  SquareMatrix() : _edgeSize(0), data() {}

  SquareMatrix(size_t edgeSize)
      : _edgeSize(edgeSize), data(edgeSize * edgeSize) {}

  SquareMatrix(const std::vector<data_t>& data)
      : _edgeSize(std::sqrt(data.size())), data(data) {
    assert(checkSizeMatch());
  }

  SquareMatrix(std::initializer_list<data_t> data)
      : _edgeSize(std::sqrt(data.size())), data(data) {
    assert(checkSizeMatch());
  }

  void updateSize() {
    if (_edgeSize * _edgeSize != data.size()) {
      _edgeSize = std::sqrt(data.size());
      assert(_edgeSize * _edgeSize == data.size());
    }
  }

  size_t edgeSize() const { return _edgeSize; }

  bool checkSizeMatch() const { return data.size() == _edgeSize * _edgeSize; }

  // get data by specifying row and col
  const data_t& getRC(size_t row, size_t col) const {
    assert(row * _edgeSize + col < data.size());
    return data[row * _edgeSize + col];
  }

  // get data by specifying row and col
  data_t& getRC(size_t row, size_t col) {
    assert(row * _edgeSize + col < data.size());
    return data[row * _edgeSize + col];
  }

  // static SquareMatrix Identity(size_t edgeSize) {
  //     SquareMatrix m(edgeSize);
  //     for (size_t r = 0; r < edgeSize; r++)
  //         m.data[r*edgeSize + r].real = 1;
  //     return m;
  // }

  // SquareMatrix matmul(const SquareMatrix& other) {
  //     assert(_edgeSize == other.edgeSize());

  //     SquareMatrix m(_edgeSize);
  //     for (size_t i = 0; i < _edgeSize; i++) {
  //     for (size_t j = 0; j < _edgeSize; j++) {
  //     for (size_t k = 0; k < _edgeSize; k++) {
  //         // C_{ij} = A_{ik} B_{kj}
  //         m.data[i*_edgeSize + j] += data[i*_edgeSize + k]* 
  //         other.data[k*_edgeSize + j];
  //     } } }
  //     return m;
  // }

  // SquareMatrix kron(const SquareMatrix& other) const {
  //     size_t lsize = size;
  //     size_t rsize = other.size;
  //     size_t msize = lsize * rsize;
  //     SquareMatrix m(msize);
  //     for (size_t lr = 0; lr < lsize; lr++) {
  //     for (size_t lc = 0; lc < lsize; lc++) {
  //     for (size_t rr = 0; rr < rsize; rr++) {
  //     for (size_t rc = 0; rc < rsize; rc++) {
  //         size_t r = lr * rsize + rr;
  //         size_t c = lc * rsize + rc;
  //         m.data[r*msize + c] = data[lr*lsize + lc] * other.data[rr*rsize +
  //         rc];
  //     } } } }
  //     return m;
  // }

  // SquareMatrix leftKronI() const {
  //     SquareMatrix m(size * size);
  //     for (size_t i = 0; i < size; i++) {
  //     for (size_t r = 0; r < size; r++) {
  //     for (size_t c = 0; c < size; c++) {
  //         m.data[(i*size + r) * size * size + (i*size + c)] = data[r*size +
  //         c];
  //     } } }
  //     return m;
  // }

  // SquareMatrix rightKronI() const {
  //     SquareMatrix m(size * size);
  //     for (size_t i = 0; i < size; i++) {
  //     for (size_t r = 0; r < size; r++) {
  //     for (size_t c = 0; c < size; c++) {
  //         m.data[(r*size + i) * size * size + (c*size + i)] = data[r*size +
  //         c];
  //     } } }
  //     return m;
  // }

  // SquareMatrix swapTargetQubits() const {
  //     assert(size == 4);
  //     return {{data[ 0], data[ 2], data[ 1], data[ 3],
  //              data[ 8], data[10], data[ 9], data[11],
  //              data[ 4], data[ 6], data[ 5], data[ 7],
  //              data[12], data[14], data[13], data[15]}};
  // }

  SquareMatrix permute(const std::vector<int>& flags) const {
    assert(utils::isPermutation(flags));
    assert(checkSizeMatch());
    assert(1 << flags.size() == _edgeSize);

    if (std::all_of(flags.begin(), flags.end(),
                    [&flags](int i) { return flags[i] == i; }))
      return SquareMatrix(*this);

    const auto permuteIndex = [&flags, k = flags.size()](size_t idx) -> size_t {
      size_t newIdx = 0;
      for (unsigned b = 0; b < k; b++) {
        newIdx += ((idx & (1ULL << b)) >> b) << flags[b];
      }
      return newIdx;
    };

    SquareMatrix matrix(_edgeSize);
    for (size_t r = 0; r < _edgeSize; r++) {
      for (size_t c = 0; c < _edgeSize; c++) {
        auto idxNew = permuteIndex(r) * _edgeSize + permuteIndex(c);
        auto idxOld = r * _edgeSize + c;
        matrix.data[idxNew] = data[idxOld];
      }
    }
    assert(matrix.checkSizeMatch());
    return matrix;
  }
};

} // namespace saot::complex_matrix

#endif // SAOT_COMPLEX_MATRIX_H