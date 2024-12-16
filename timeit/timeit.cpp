#include "timeit/timeit.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace timeit;

namespace {
double getMedian(const std::vector<double>& arr, size_t start, size_t end) {
  auto l = end - start;
  if (l % 2 == 0)
    return 0.5 * (arr[start + l / 2 - 1] + arr[start + l / 2]);
  return arr[start + l / 2];
}
} // namespace

void TimingResult::calcStats() {
  auto l = tArr.size();
  if (repeat == 0) {
    return;
  }
  if (l == 0) {
    return;
  }
  if (l == 1) {
    min = tArr[0];
    med = tArr[0];
    return;
  }
  std::sort(tArr.begin(), tArr.end());
  min = tArr[0] / repeat;
  med = getMedian(tArr, 0, tArr.size()) / repeat;
  q1 = getMedian(tArr, 0, tArr.size() / 2) / repeat;
  q3 = getMedian(tArr, tArr.size() / 2, tArr.size()) / repeat;
}

std::string TimingResult::timeToString(double t, int n_sig_dig) {
  double timeValue;
  std::string unit;

  if (t >= 1) {
    timeValue = t;
    unit = "s";
  } else if (t >= 1e-3) {
    timeValue = t * 1e3;
    unit = "ms";
  } else if (t >= 1e-6) {
    timeValue = t * 1e6;
    unit = "us";
  } else {
    timeValue = t * 1e9;
    unit = "ns";
  }

  // significant digits
  int precision =
      n_sig_dig - 1 - static_cast<int>(std::floor(std::log10(timeValue)));

  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << timeValue << " "
         << unit;
  return stream.str();
}

std::ostream& TimingResult::display(int n_sig_dig, std::ostream& os) const {
  os << replication << " replications (" << repeat << " repeats each): "
     << "min " << timeToString(min, n_sig_dig) << "; median "
     << timeToString(med, n_sig_dig) << "\n";
  return os;
}

std::string TimingResult::raw_string() const {
  if (tArr.size() == 0) {
    return "[]";
  }
  std::stringstream ss;
  ss << "[" << std::scientific;
  for (size_t i = 0; i < tArr.size() - 1; ++i) {
    ss << (tArr[i] / repeat) << " ";
  }
  ss << (tArr[tArr.size() - 1] / repeat) << "]";
  return ss.str();
}

TimingResult Timer::timeit(std::function<void()> method,
                           std::function<void()> setup,
                           std::function<void()> teardown) {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  unsigned repeat = 1;
  std::vector<double> tarr(replication);
  TimePoint tic, toc;
  double dur;

  if (verbose > 0) {
    std::cerr << "Warm-up time: " << warmupTime << " s;\n";
    std::cerr << "Running time: " << runTime << " s;\n";
  }

  setup();

  // warmup
  tic = Clock::now();
  method();
  toc = Clock::now();
  dur = Duration(toc - tic).count();

  unsigned r0 = 0;
  if (dur > warmupTime) {
    tarr[0] = dur;
    r0 = 1;
  } else {
    repeat = static_cast<double>(repeat) * warmupTime / dur + 1;
    tic = Clock::now();
    for (unsigned i = 0; i < repeat; ++i)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
  }

  // main loop
  repeat = static_cast<double>(repeat) * runTime / dur + 1;
  tarr[0] *= repeat;
  for (unsigned r = r0; r < replication; ++r) {
    tic = Clock::now();
    for (unsigned i = 0; i < repeat; ++i)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
    tarr[r] = dur;
  }

  teardown();

  return TimingResult(repeat, replication, tarr);
}

TimingResult Timer::timeitFixedRepeat(std::function<void()> method, int _repeat,
                                      std::function<void()> setup,
                                      std::function<void()> teardown) {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  std::vector<double> tarr(replication);
  TimePoint tic, toc;
  double dur;

  setup();
  // main loop
  for (unsigned r = 0; r < replication; ++r) {
    tic = Clock::now();
    for (unsigned i = 0; i < _repeat; ++i)
      method();
    toc = Clock::now();
    dur = Duration(toc - tic).count();
    tarr[r] = dur;
  }

  teardown();

  return TimingResult(_repeat, replication, tarr);
}