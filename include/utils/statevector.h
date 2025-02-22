#ifndef UTILS_STATEVECTOR_H
#define UTILS_STATEVECTOR_H

#include <complex>
#include <iostream>
#include <random>
#include <thread>
#include <algorithm>
#include <cstdlib>

#include "cast/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

namespace utils {

template<typename real_t>
class StatevectorSep;

template<typename real_t>
class StatevectorAlt;

template<typename real_t>
class StatevectorSep {
public:
  int nQubits;
  uint64_t N;
  real_t* real;
  real_t* imag;

  StatevectorSep(int nQubits, bool initialize = false)
      : nQubits(nQubits), N(1ULL << nQubits) {
    assert(nQubits > 0);
    real = (real_t*)std::aligned_alloc(64, N * sizeof(real_t));
    imag = (real_t*)std::aligned_alloc(64, N * sizeof(real_t));
    if (initialize) {
      for (size_t i = 0; i < (1 << nQubits); i++) {
        real[i] = 0;
        imag[i] = 0;
      }
      real[0] = 1.0;
    }
    // std::cerr << "StatevectorSep(int)\n";
  }

  StatevectorSep(const StatevectorSep& that)
      : nQubits(that.nQubits), N(that.N) {
    real = (real_t*)aligned_alloc(64, N * sizeof(real_t));
    imag = (real_t*)aligned_alloc(64, N * sizeof(real_t));
    for (size_t i = 0; i < that.N; i++) {
      real[i] = that.real[i];
      imag[i] = that.imag[i];
      // std::cerr << "StatevectorSep(const StatevectorSep&)\n";
    }
  }

  StatevectorSep(StatevectorSep&& that) noexcept
      : nQubits(that.nQubits), N(that.N), real(that.real), imag(that.imag) {
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

  StatevectorSep& operator=(StatevectorSep&& that) noexcept {
    this->~StatevectorSep();
    real = that.real;
    imag = that.imag;
    nQubits = that.nQubits;
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
/// memory index: 000 001 010 011 100 101 110 111
/// amplitudes:   r00 r01 i00 i01 r10 r11 i10 i11
template<typename RealType>
class StatevectorAlt {
public:
  int simd_s;
  int nQubits;
  uint64_t N;
  size_t memSize;
  RealType* data;

  StatevectorAlt(int nQubits, int simd_s, bool init = true)
    : simd_s(simd_s)
    , nQubits(nQubits)
    , N(1ULL << nQubits)
    , memSize((2ULL << nQubits) * sizeof(RealType))
    , data(static_cast<RealType*>(
        ::operator new(memSize, static_cast<std::align_val_t>(64)))) {
    if (init)
      initialize();
  }

  StatevectorAlt(const StatevectorAlt&) = delete;

  StatevectorAlt(StatevectorAlt&&) = delete;

  ~StatevectorAlt() { ::operator delete(data); }

  StatevectorAlt& operator=(const StatevectorAlt& that) {
    if (this == &that)
      return *this;
    std::memcpy(data, that.data, memSize);
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

  /// @brief Initialize to the |00...00> state.
  void initialize() {
    std::memset(data, 0, memSize);
    data[0] = 1.0;
  }

  void normalize() {
    double n = norm();
    for (size_t i = 0; i < 2 * N; i++)
      data[i] /= n;
  }

  /// @brief Uniform randomize statevector (by the Haar-measure on sphere).
  void randomize() {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<RealType> d{0, 1};
    for (size_t i = 0; i < 2 * N; i++)
      data[i] = d(gen);
    normalize();
  }

  RealType& real(size_t idx) {
    return data[utils::insertZeroToBit(idx, simd_s)];
  }
  RealType& imag(size_t idx) {
    return data[utils::insertOneToBit(idx, simd_s)];
  }
  const RealType& real(size_t idx) const {
    return data[utils::insertZeroToBit(idx, simd_s)];
  }
  const RealType& imag(size_t idx) const {
    return data[utils::insertOneToBit(idx, simd_s)];
  }

  std::complex<RealType> amp(size_t idx) const {
    size_t tmp = utils::insertZeroToBit(idx, simd_s);
    return { data[tmp], data[tmp | (1 << simd_s)] };
  }

  std::ostream& print(std::ostream& os = std::cerr) const {
    if (N > 32) {
      os << BOLDCYAN("Warning: ")
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < std::min<uint64_t>(32ULL, N); i++) {
      os << i << ": ";
      utils::print_complex(os, {real(i), imag(i)}, 8);
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
    for (int q = 0; q < nQubits; q++) {
      os << "qubit " << q << ": " << prob(q) << "\n";
    }
    return os;
  }

  StatevectorAlt& applyGate(const cast::QuantumGate& gate) {
    const auto* cMat = gate.gateMatrix.getConstantMatrix();
    assert(cMat && "Can only apply constant gateMatrix");

    const unsigned k = gate.qubits.size();
    const unsigned K = 1 << k;
    assert(cMat->edgeSize() == K);
    std::vector<size_t> ampIndices(K);
    std::vector<std::complex<RealType>> ampUpdated(K);

    size_t pdepMaskTask = ~static_cast<size_t>(0);
    size_t pdepMaskAmp = 0;
    for (const auto& q : gate.qubits) {
      pdepMaskTask ^= (1ULL << q);
      pdepMaskAmp |= (1ULL << q);
    }

    for (size_t taskId = 0; taskId < (N >> k); taskId++) {
      auto pdepTaskId = utils::pdep64(taskId, pdepMaskTask);
      for (size_t ampId = 0; ampId < K; ampId++) {
        ampIndices[ampId] = pdepTaskId + utils::pdep64(ampId, pdepMaskAmp);
      }

      // std::cerr << "taskId = " << taskId
      //           << " (" << utils::as0b(taskId, nQubits - k) << "):\n";
      // utils::printVectorWithPrinter(ampIndices,
      //   [&](size_t n, std::ostream& os) {
      //     os << n << " (" << utils::as0b(n, nQubits) << ")";
      //   }, std::cerr << " ampIndices: ") << "\n";

      for (unsigned r = 0; r < K; r++) {
        ampUpdated[r] = 0.0;
        for (unsigned c = 0; c < K; c++) {
          ampUpdated[r] += cMat->rc(r, c) * this->amp(ampIndices[c]);
        }
      }
      for (unsigned r = 0; r < K; r++) {
        this->real(ampIndices[r]) = ampUpdated[r].real();
        this->imag(ampIndices[r]) = ampUpdated[r].imag();
      }
    }
    return *this;
  }

}; // class StatevectorAlt

template<typename real_t>
double fidelity(const StatevectorSep<real_t>& sv1,
                       const StatevectorSep<real_t>& sv2) {
  assert(sv1.nQubits == sv2.nQubits);

  double re = 0.0, im = 0.0;
  for (size_t i = 0; i < sv1.N; i++) {
    re += (sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
    im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
  }
  return re * re + im * im;
}

template<typename real_t>
double fidelity(const StatevectorAlt<real_t>& sv0,
                const StatevectorAlt<real_t>& sv1) {
  assert(sv0.nQubits == sv1.nQubits);
  double re = 0.0, im = 0.0;
  for (size_t i = 0; i < sv0.N; i++) {
    auto amp0 = sv0.amp(i);
    auto amp1 = sv1.amp(i);
    re += amp0.real() * amp1.real() + amp0.imag() * amp1.imag();
    im += amp0.real() * amp1.imag() - amp0.imag() * amp1.real();
  }
  return re * re + im * im;
}

// template<typename real_t>
// static double fidelity(const StatevectorSep<real_t>& sep, const
// StatevectorAlt<real_t>& alt) {
//     assert(sep.nQubits == alt.nQubits);

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
//     assert(nQubits == alt.nQubits);

//     for (size_t i = 0; i < N; i++) {
//         real[i] = alt.data[2*i];
//         imag[i] = alt.data[2*i+1];
//     }
// }

// template<typename real_t>
// void StatevectorAlt<real_t>::copyValueFrom(const StatevectorSep<real_t>& sep)
// {
//     assert(nQubits == sep.nQubits);

//     for (size_t i = 0; i < N; i++) {
//         data[2*i] = sep.real[i];
//         data[2*i+1] = sep.imag[i];
//     }
// }

} // namespace utils::statevector

#endif // UTILS_STATEVECTOR_H