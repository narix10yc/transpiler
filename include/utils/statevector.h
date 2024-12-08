#ifndef UTILS_STATEVECTOR_H
#define UTILS_STATEVECTOR_H

#include <bitset>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <thread>

#include "utils/iocolor.h"
#include "utils/utils.h"

namespace utils::statevector {

template<typename real_t>
class StatevectorSep;

template<typename real_t, unsigned simd_s>
class StatevectorAlt;

template<typename real_t>
class StatevectorSep {
public:
  unsigned nqubits;
  uint64_t N;
  real_t* real;
  real_t* imag;

  StatevectorSep(int nqubits, bool initialize = false)
      : nqubits(static_cast<unsigned>(nqubits)), N(1ULL << nqubits) {
    assert(nqubits > 0);
    real = (real_t*)std::aligned_alloc(64, N * sizeof(real_t));
    imag = (real_t*)std::aligned_alloc(64, N * sizeof(real_t));
    if (initialize) {
      for (size_t i = 0; i < (1 << nqubits); i++) {
        real[i] = 0;
        imag[i] = 0;
      }
      real[0] = 1.0;
    }
    // std::cerr << "StatevectorSep(int)\n";
  }

  StatevectorSep(const StatevectorSep& that)
      : nqubits(that.nqubits), N(that.N) {
    real = (real_t*)aligned_alloc(64, N * sizeof(real_t));
    imag = (real_t*)aligned_alloc(64, N * sizeof(real_t));
    for (size_t i = 0; i < that.N; i++) {
      real[i] = that.real[i];
      imag[i] = that.imag[i];
      // std::cerr << "StatevectorSep(const StatevectorSep&)\n";
    }
  }

  StatevectorSep(StatevectorSep&&that)
      : nqubits(that.nqubits), N(that.N), real(that.real), imag(that.imag) {
    that.real = nullptr;
    that.imag = nullptr;
    // std::cerr << "StatevectorSep(StatevectorSep&&)\n";
  }

  ~StatevectorSep() {
    std::free(real);
    std::free(imag);
    // std::cerr << "~StatevectorSep\n";
  }

  StatevectorSep& operator=(const StatevectorSep& that) {
    if (this != &that) {
      for (size_t i = 0; i < N; i++) {
        real[i] = that.real[i];
        imag[i] = that.imag[i];
      }
    }
    // std::cerr << "=(const StatevectorSep&)\n";
    return *this;
  }

  StatevectorSep& operator=(StatevectorSep&&that) {
    this->~StatevectorSep();
    real = that.real;
    imag = that.imag;
    nqubits = that.nqubits;
    N = that.N;

    that.real = nullptr;
    that.imag = nullptr;
    // std::cerr << "=(StatevectorSep&&)\n";
    return *this;
  }

  // void copyValueFrom(const StatevectorAlt<real_t>&);

  double normSquared(int nthreads = 1) const {
    const auto f = [&](uint64_t i0, uint64_t i1, double &rst) {
      double sum = 0.0;
      for (uint64_t i = i0; i < i1; i++) {
        sum += real[i] * real[i];
        sum += imag[i] * imag[i];
      }
      rst = sum;
    };

    if (nthreads == 1) {
      double s;
      f(0, N, s);
      return s;
    }

    std::vector<std::thread> threads(nthreads);
    std::vector<double> sums(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1, std::ref(sums[i]));
    }

    for (auto& thread : threads)
      thread.join();

    double sum = 0.0;
    for (const auto& s : sums)
      sum += s;
    return sum;
  }

  double norm(int nthreads = 1) const {
    return std::sqrt(normSquared(nthreads));
  }

  void normalize(int nthreads = 1) {
    double n = norm(nthreads);
    const auto f = [&](uint64_t i0, uint64_t i1) {
      for (uint64_t i = i0; i < i1; i++) {
        real[i] /= n;
        imag[i] /= n;
      }
    };

    if (nthreads == 1) {
      f(0, N);
      return;
    }
    std::vector<std::thread> threads(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1);
    }

    for (auto& thread : threads)
      thread.join();
  }

  void randomize(int nthreads = 1) {
    const auto f = [&](uint64_t i0, uint64_t i1) {
      std::random_device rd;
      std::mt19937 gen{rd()};
      std::normal_distribution<real_t> d{0, 1};
      for (uint64_t i = i0; i < i1; i++) {
        real[i] = d(gen);
        imag[i] = d(gen);
      }
    };

    if (nthreads == 1) {
      f(0, N);
      normalize(nthreads);
      return;
    }

    std::vector<std::thread> threads(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1);
    }

    for (auto& thread : threads)
      thread.join();
    normalize(nthreads);
  }

  std::ostream& print(std::ostream& os) const {
    if (N > 32) {
      os << IOColor::BOLD << IOColor::CYAN_FG << "Warning: " << IOColor::RESET
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
      os << i << ": ";
      utils::print_complex(os, {real[i], imag[i]});
      os << "\n";
    }
    return os;
  }
};


/// @brief StatevectorAlt stores statevector in a single array with alternating
/// real and imaginary parts. The alternating pattern is controlled by
/// \p simd_s. More precisely, the memory is stored as an iteration of $2^s$
/// real parts followed by $2^s$ imaginary parts.
/// For example,
/// 000 001 010 011 100 101 110 111
/// r00 r01 i00 i01 r10 r11 i10 i11
template<typename real_t, unsigned simd_s>
class StatevectorAlt {
public:
  int nqubits;
  uint64_t N;
  size_t memSize;
  real_t* data;

  StatevectorAlt(int nqubits, bool init = true)
      : nqubits(nqubits), N(1ULL << nqubits)
      , memSize((2ULL << nqubits) * sizeof(real_t)) {
    data = (real_t*)aligned_alloc(64, this->memSize);
    assert(data && "Allocation Failed");
    if (init)
      initialize();
  }

  StatevectorAlt(const StatevectorAlt& that)
      : nqubits(that.nqubits), N(that.N), memSize(that.memSize) {
    data = (real_t*)aligned_alloc(64, memSize);
    std::memcpy(data, that.data, memSize);
  }

  StatevectorAlt(StatevectorAlt&&) = delete;

  ~StatevectorAlt() { std::free(data); }

  StatevectorAlt& operator=(const StatevectorAlt& that) {
    if (this != &that) {
      for (size_t i = 0; i < 2 * N; i++)
        data[i] = that.data[i];
    }
    return *this;
  }

  StatevectorAlt& operator=(StatevectorAlt&&) = delete;

  double normSquared() const {
    double s = 0;
    for (size_t i = 0; i < 2 * N; i++)
      s += data[i] * data[i];
    return s;
  }

  double norm() const { return std::sqrt(normSquared()); }

  void initialize() {
    std::memset(data, 0, memSize);
    data[0] = 1.0;
  }

  void normalize() {
    double n = norm();
    for (size_t i = 0; i < 2 * N; i++)
      data[i] /= n;
  }

  void randomize() {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<real_t> d{0, 1};
    for (size_t i = 0; i < 2 * N; i++)
      data[i] = d(gen);
    normalize();
  }

  real_t& real(size_t idx) {
    return data[utils::insertZeroToBit(idx, simd_s)];
  }
  real_t& imag(size_t idx) {
    return data[utils::insertOneToBit(idx, simd_s)];
  }
  const real_t& real(size_t idx) const {
    return data[utils::insertZeroToBit(idx, simd_s)];
  }
  const real_t& imag(size_t idx) const {
    return data[utils::insertOneToBit(idx, simd_s)];
  }

  std::ostream& print(std::ostream& os = std::cerr) const {
    if (N > 32) {
      os << BOLDCYAN("Warning: ")
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < std::min(32ULL, N); i++) {
      os << i << ": ";
      utils::print_complex(os, {real(i), imag(i)});
      os << "\n";
    }
    return os;
  }

  /// @brief Compute the probability of measuring 1 on qubit q
  double prob(int q) const {
    double p = 0.0;
    for (size_t i = 0; i < (N >> 1); i++) {
      size_t idx = utils::insertZeroToBit(i, q);
      const auto& re = real(idx);
      const auto& im = imag(idx);
      p += (re * re + im * im);
    }
    return 1 - p;
  }

  std::ostream& printProbabilities(std::ostream& os = std::cerr) const {
    for (int q = 0; q < nqubits; q++) {
      os << "qubit " << q << ": " << prob(q) << "\n";
    }
    return os;
  }

}; // class StatevectorAlt

template<typename real_t>
static double fidelity(const StatevectorSep<real_t>& sv1,
                       const StatevectorSep<real_t>& sv2) {
  assert(sv1.nqubits == sv2.nqubits);

  double re = 0.0, im = 0.0;
  for (size_t i = 0; i < sv1.N; i++) {
    re += (sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
    im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
  }
  return re * re + im * im;
}

// template<typename real_t>
// static double fidelity(const StatevectorSep<real_t>& sep, const
// StatevectorAlt<real_t>& alt) {
//     assert(sep.nqubits == alt.nqubits);

//     double re = 0.0, im = 0.0;
//     for (size_t i = 0; i < sep.N; i++) {
//         re += ( sep.real[i] * alt.data[2*i] + sep.imag[i] * alt.data[2*i+1]);
//         im += (-sep.real[i] * alt.data[2*i+1] + sep.imag[i] * alt.data[2*i]);
//     }
//     return re * re + im * im;
// }

// template<typename real_t>
// double fidelity(const StatevectorAlt<real_t>& alt, const
// StatevectorSep<real_t>& sep) {
//     return fidelity(sep, alt);
// }

// template<typename real_t>
// void StatevectorSep<real_t>::copyValueFrom(const StatevectorAlt<real_t>& alt)
// {
//     assert(nqubits == alt.nqubits);

//     for (size_t i = 0; i < N; i++) {
//         real[i] = alt.data[2*i];
//         imag[i] = alt.data[2*i+1];
//     }
// }

// template<typename real_t>
// void StatevectorAlt<real_t>::copyValueFrom(const StatevectorSep<real_t>& sep)
// {
//     assert(nqubits == sep.nqubits);

//     for (size_t i = 0; i < N; i++) {
//         data[2*i] = sep.real[i];
//         data[2*i+1] = sep.imag[i];
//     }
// }

} // namespace utils::statevector

#endif // UTILS_STATEVECTOR_H