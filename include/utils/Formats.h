#ifndef UTILS_FORMATS_H
#define UTILS_FORMATS_H

#include <iostream>
#include <cstdlib>

namespace utils {

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



} // namespace utils

#endif // UTILS_FORMATS_H