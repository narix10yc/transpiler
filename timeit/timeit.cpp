#include "timeit/timeit.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace timeit;

namespace {
double getMedian(const std::vector<double>& arr, size_t start, size_t end) {
    auto l = end - start;
    if (l % 2 == 0)
        return 0.5 * (arr[start + l/2 - 1] + arr[start + l/2]);
    return arr[start + l/2];
}
} // <anonymous> namespace

void TimingResult::calcStats() {
    auto l = tarr.size();
    if (repeat == 0) { return; }
    if (l == 0) { return; }
    if (l == 1) {
        min = tarr[0];
        med = tarr[0];
        return;
    }
    std::sort(tarr.begin(), tarr.end());
    min = tarr[0];
    med = getMedian(tarr, 0, tarr.size());
    
    min /= repeat;
    med /= repeat;
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
    int precision = n_sig_dig - 1 - static_cast<int>(std::floor(std::log10(timeValue)));

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << timeValue << " " << unit;
    return stream.str();
}



void TimingResult::display(int n_sig_dig) const {
    std::cout << replication << " replications (" << repeat << " repeats each): ";
    std::cout << "min " << timeToString(min, n_sig_dig)
              << "; median " << timeToString(med, n_sig_dig);
    std::cout << std::endl;
}

std::string TimingResult::raw_string() const {
    if (tarr.size() == 0) { return "[]"; }
    std::stringstream ss;
    ss << "[" << std::scientific;
    for (size_t i = 0; i < tarr.size() - 1; ++i) {
        ss << (tarr[i] / repeat) << " ";
    }
    ss << (tarr[tarr.size()-1] / repeat) << "]";
    return ss.str();
}

TimingResult Timer::timeit(
    std::function<void()> method, 
    std::function<void()> setup,
    std::function<void()> teardown
    ) {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;

    size_t repeat = 1;
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

    while (dur < warmupTime) {
        repeat = (double)repeat * warmupTime / dur;
        ++repeat;
        tic = Clock::now();
        for (size_t i = 0; i < repeat; ++i)
            method();
        toc = Clock::now();
        dur = Duration(toc - tic).count();
    }

    // main loop
    repeat = (double)repeat * runTime / dur; 
    ++repeat;
    for (size_t r = 0; r < replication; ++r)
    {   
        tic = Clock::now();
        for (size_t i = 0; i < repeat; ++i)
            method();
        toc = Clock::now();
        dur = Duration(toc - tic).count();
        tarr[r] = dur;
    }

    teardown();

    return TimingResult(repeat, replication, tarr);
}