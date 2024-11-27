#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <llvm/ADT/SmallVector.h>

namespace utils {

bool isPermutation(const std::vector<int> &v);

template <typename T>
static bool isOrdered(const std::vector<T> &vec, bool ascending = true) {
  if (vec.empty())
    return true;

  if (ascending) {
    for (unsigned i = 0; i < vec.size() - 1; i++) {
      if (vec[i] > vec[i + 1])
        return false;
    }
    return true;
  } else {
    for (unsigned i = 0; i < vec.size() - 1; i++) {
      if (vec[i] < vec[i + 1])
        return false;
    }
    return true;
  }
}

std::ostream &print_complex(std::ostream &os, std::complex<double> c,
                            int precision = 3);

template <typename T>
std::ostream &printVector(const std::vector<T> &v,
                          std::ostream &os = std::cerr) {
  if (v.empty())
    return os << "[]";
  auto it = v.cbegin();
  os << "[" <<* it;
  while (++it != v.cend())
    os << "," <<* it;
  return os << "]";
}

template <typename T, unsigned N>
std::ostream &printVector(const llvm::SmallVector<T, N> &v,
                          std::ostream &os = std::cerr) {
  if (v.empty())
    return os << "[]";
  auto it = v.cbegin();
  os << "[" <<* it;
  while (++it != v.cend())
    os << "," <<* it;
  return os << "]";
}

// The printer is expected to take inputs (const T&, std::ostream&)
template <typename T, typename Printer_T>
std::ostream &printVectorWithPrinter(const std::vector<T> &v, Printer_T f,
                                     std::ostream &os = std::cerr) {
  if (v.empty())
    return os << "[]";
  auto it = v.cbegin();
  f(*it, os << "[");
  while (++it != v.cend())
    f(*it, os << ",");
  return os << "]";
}

// @return true if elem is in vec
template <typename T>
static void pushBackIfNotInVector(std::vector<T> &vec, T elem) {
  for (const auto &e : vec) {
    if (e == elem)
      return;
  }
  vec.push_back(elem);
}

template <typename T = uint64_t> static T insertZeroToBit(T x, int bit) {
  T maskLo = (1 << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) + ((x & maskHi) << 1);
}

template <typename T = uint64_t> static T insertOneToBit(T x, int bit) {
  T maskLo = (1 << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) | ((x & maskHi) << 1) | (1 << bit);
}

uint64_t pdep64(uint64_t src, uint64_t mask);
uint64_t pdep64(uint64_t src, uint64_t mask, int nbits);

uint32_t pdep32(uint32_t src, uint32_t mask);
uint32_t pdep32(uint32_t src, uint32_t mask, int nbits);

uint64_t pext64(uint64_t src, uint64_t mask);
uint64_t pext64(uint64_t src, uint64_t mask, int nbits);

class as0b {
  uint64_t v;
  int nbits;

public:
  as0b(uint64_t v, int nbits) : v(v), nbits(nbits) {
    assert(nbits > 0 && nbits <= 64);
  }

  friend std::ostream &operator<<(std::ostream &os, const as0b &n) {
    for (int i = n.nbits - 1; i >= 0; --i)
      os.put((n.v & (1 << i)) ? '1' : '0');
    return os;
  }
};

void timedExecute(std::function<void()> f, const char* msg);

} // namespace utils

#endif // UTILS_UTILS_H