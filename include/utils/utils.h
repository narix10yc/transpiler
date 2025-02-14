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
#include <span>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"

namespace utils {

bool isPermutation(llvm::ArrayRef<int> arr);

template<typename T>
bool isOrdered(const std::vector<T>& vec, bool ascending = true) {
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

std::ostream&
print_complex(std::ostream& os, std::complex<double> c, int precision = 3);

template<typename T>
std::ostream& printArrayNoBrackets(
    std::ostream& os, llvm::ArrayRef<T> arr, const char sep = ',') {
  if (arr.empty())
    return os;
  auto it = arr.begin();
  os << *it;
  while (++it != arr.end()) {
    os.put(sep);
    os << *it;
  }
  return os;
}

template<typename T>
std::ostream& printArray(
    std::ostream& os, llvm::ArrayRef<T> arr, const char sep = ',') {
  if (arr.empty())
    return os << "[]";
  auto it = arr.begin();
  os << "[" << *it;
  while (++it != arr.end()) {
    os.put(sep);
    os << *it;
  }
  return os << "]";
}

template<typename T, typename Printer>
std::ostream& printArray(
    std::ostream& os, llvm::ArrayRef<T> arr,
    Printer pFunc,
    const char sep = ',') {
  if (arr.empty())
    return os << "[]";
  auto it = arr.begin();
  os << "[";
  pFunc(os, *it);
  while (++it != arr.end()) {
    os.put(sep);
    pFunc(os, *it);
  }
  return os << "]";
}

template<typename T, unsigned N>
std::ostream& printArray(
    std::ostream& os, const llvm::SmallVector<T, N>& arr, const char sep = ',') {
  return printArray(os, llvm::ArrayRef<T>(arr), sep);
}

// @param f: The printer is expected to take inputs (const T&, std::ostream&)
template<typename T, typename Printer_T>
std::ostream& printVectorWithPrinter(
    const std::vector<T>& v, Printer_T f, std::ostream& os = std::cerr) {
  if (v.empty())
    return os << "[]";
  auto it = v.cbegin();
  f(*it, os << "[");
  while (++it != v.cend())
    f(*it, os << ",");
  return os << "]";
}

template<typename T>
void pushBackIfNotInVector(std::vector<T>& vec, T elem) {
  for (const auto& e : vec) {
    if (e == elem)
      return;
  }
  vec.push_back(elem);
}

template<typename T, unsigned N>
void pushBackIfNotInVector(llvm::SmallVector<T, N>& vec, T elem) {
  for (const auto& e : vec) {
    if (e == elem)
      return;
  }
  vec.push_back(elem);
}


template<typename T = uint64_t>
T insertZeroToBit(T x, int bit) {
  T maskLo = (static_cast<T>(1) << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) + ((x & maskHi) << 1);
}

template<typename T = uint64_t>
T insertOneToBit(T x, int bit) {
  T maskLo = (static_cast<T>(1) << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) | ((x & maskHi) << 1) | (1 << bit);
}

// parallel bit deposition
uint64_t pdep64(uint64_t src, uint64_t mask, int nbits = 64);
uint32_t pdep32(uint32_t src, uint32_t mask, int nbits = 32);

// parallel bit extraction
uint64_t pext64(uint64_t src, uint64_t mask, int nbits = 64);
uint32_t pext32(uint32_t src, uint32_t mask, int nbits = 32);


// as0b: helper class to print binary of an integer
// uses: 'std::cerr << as0b(123, 12)' to print 12 LSB of integer 123.
class as0b {
  uint64_t v;
  int nbits;

public:
  as0b(uint64_t v, int nbits) : v(v), nbits(nbits) {
    assert(nbits >= 0 && nbits <= 64);
  }

  friend std::ostream& operator<<(std::ostream& os, const as0b& n) {
    for (int i = n.nbits - 1; i >= 0; --i)
      os.put((n.v & (1 << i)) ? '1' : '0');
    return os;
  }
};

class fmt_time {
  double t_in_sec;
public:
  explicit fmt_time(double t_in_sec) : t_in_sec(t_in_sec) {
    assert(t_in_sec >= 0.0);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_time& fmt) {
    // seconds
    if (fmt.t_in_sec >= 1e3)
      return os << static_cast<unsigned>(fmt.t_in_sec) << " s";
    if (fmt.t_in_sec >= 1e2)
      return os << std::fixed << std::setprecision(1)
                << fmt.t_in_sec << " s";
    if (fmt.t_in_sec >= 1e1)
      return os << std::fixed << std::setprecision(2)
                << fmt.t_in_sec << " s";
    if (fmt.t_in_sec >= 1.0)
      return os << std::fixed << std::setprecision(3)
                << fmt.t_in_sec << " s";
    // milliseconds
    if (fmt.t_in_sec >= 1e-1)
      return os << std::fixed << std::setprecision(1)
                << 1e3 * fmt.t_in_sec << " ms";
    if (fmt.t_in_sec >= 1e-2)
      return os << std::fixed << std::setprecision(2)
                << 1e3 * fmt.t_in_sec << " ms";
    if (fmt.t_in_sec >= 1e-3)
      return os << std::fixed << std::setprecision(3)
                << 1e3 * fmt.t_in_sec << " ms";
    // microseconds
    if (fmt.t_in_sec >= 1e-4)
      return os << std::fixed << std::setprecision(1)
                << 1e6 * fmt.t_in_sec << " us";
    if (fmt.t_in_sec >= 1e-5)
      return os << std::fixed << std::setprecision(2)
                << 1e6 * fmt.t_in_sec << " us";
    if (fmt.t_in_sec >= 1e-6)
      return os << std::fixed << std::setprecision(3)
                << 1e6 * fmt.t_in_sec << " us";
    // nanoseconds
    if (fmt.t_in_sec >= 1e-7)
      return os << std::fixed << std::setprecision(1)
                << 1e9 * fmt.t_in_sec << " ns";
    if (fmt.t_in_sec >= 1e-8)
      return os << std::fixed << std::setprecision(2)
                << 1e9 * fmt.t_in_sec << " ns";
    if (fmt.t_in_sec >= 1e-9)
      return os << std::fixed << std::setprecision(3)
                << 1e9 * fmt.t_in_sec << " ns";
    return os << "< 1.0 ns";
  }
};

class fmt_1_to_1e3 {
  double number;
  int width;

public:
  explicit fmt_1_to_1e3(double n, int width) : number(n), width(width) {
    assert(n >= 0.0 && "fmt_1_to_1e3: Currently only supporting positive numbers");
    assert(width >= 4);
    // assert(n >= 1.0 && n <= 1e3);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_1_to_1e3& fmt) {
    if (fmt.number >= 100.0)
      return os << std::fixed << std::setprecision(fmt.width - 4) << fmt.number;
    if (fmt.number >= 10.0)
      return os << std::fixed << std::setprecision(fmt.width - 3) << fmt.number;
    return os << std::fixed << std::setprecision(fmt.width - 2) << fmt.number;
  }
};

inline void displayProgressBar(float progress, int barWidth = 50) {
  // Clamp progress between 0 and 1
  assert(barWidth > 0);
  if (progress < 0.0f) progress = 0.0f;
  if (progress > 1.0f) progress = 1.0f;

  // Print the progress bar
  std::cout.put('[');
  int i = 0;
  while (i < barWidth * progress) {
    std::cout.put('=');
    ++i;
  }
  while (i < barWidth) {
    std::cout.put(' ');
    ++i;
  }

  std::cout << "] " << static_cast<int>(progress * 100.0f) << " %\r";
  std::cout.flush();
}

inline void displayProgressBar(int nFinished, int nTotal, int barWidth = 50) {
  return displayProgressBar(static_cast<float>(nFinished) / nTotal, barWidth);
}


void timedExecute(std::function<void()> f, const char* msg);

/// @brief a dagger dotted with b
std::complex<double> inner_product(
    const std::complex<double>* aArrBegin, 
    const std::complex<double>* bArrBegin,
    size_t length);

double norm_squared(const std::complex<double>* arrBegin, size_t length);

inline double norm(const std::complex<double>* arrBegin, size_t length) {
  return std::sqrt(norm_squared(arrBegin, length));
}

inline void normalize(std::complex<double>* arrBegin, size_t length) {
  double norm = utils::norm(arrBegin, length);
  for (size_t i = 0; i < length; i++)
    arrBegin[i] /= norm;
}

} // namespace utils

#endif // UTILS_UTILS_H