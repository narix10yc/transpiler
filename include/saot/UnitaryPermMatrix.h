#ifndef SAOT_UNITARYPERMMATRIX_H
#define SAOT_UNITARYPERMMATRIX_H

#include "utils/utils.h"

#include <vector>

namespace saot {

/// @brief A square matrix that each row or column has exactly one non-zero
/// entry of the form expi(phi).
/// @param data The (i, data[i].first) entry of the matrix is
/// expi(data[i].second)
template<typename data_t>
class UnitaryPermutationMatrix {
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

  std::pair<size_t, data_t>& operator[](size_t index) { return data[index]; }
  const std::pair<size_t, data_t>& operator[](size_t index) const {
    return data[index];
  }

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

} // namespace saot

#endif // SAOT_UNITARYPERMMATRIX_H