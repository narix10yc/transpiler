#ifndef SIMULATION_TIMEIT_H_
#define SIMULATION_TIMEIT_H_

#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <sstream>
#include <iomanip>


namespace timeit {
class TimingResult;
class Timer;

namespace {
double getMedian(const std::vector<double>& arr, size_t start, size_t end) {
    auto l = end - start;
    if (l % 2 == 0)
        return 0.5 * (arr[start + l/2 - 1] + arr[start + l/2]);
    return arr[start + l/2];
}

std::string TimingResult::timeToString(double t, int n_sig_dig) {
    double timeValue;
    std::string unit;

    if (t >= 1) {
        timeValue = t;        unit = "s";
    } else if (t >= 1e-3) {
        timeValue = t * 1e3;  unit = "ms";
    } else if (t >= 1e-6) {
        timeValue = t * 1e6;  unit = "us";
    } else {
        timeValue = t * 1e9;  unit = "ns";
    }

    // significant digits
    int precision = n_sig_dig - 1 - static_cast<int>(std::floor(std::log10(timeValue)));

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << timeValue << " " << unit;
    return stream.str();
}
} // <anonymous> namespace

class TimingResult {
size_t repeat, replication;
public:
    std::vector<double> tarr;
    double min, med, q1, q3;
    int n_sig_dig = 4;

    TimingResult() {}
    TimingResult(size_t repeat, size_t replication, std::vector<double>& tarr) 
        : repeat(repeat), replication(replication), tarr(tarr) { calcStats(); }

    static std::string timeToString(double, int);
    
    void display(int=4) const {
        std::cout << replication << " replications (" << repeat << " repeats each): ";
        std::cout << "min " << timeToString(min, n_sig_dig)
                << "; median " << timeToString(med, n_sig_dig);
        std::cout << std::endl;
    }

    std::string raw_string() const {
        if (tarr.size() == 0) { return "[]"; }
        std::stringstream ss;
        ss << "[" << std::scientific;
        for (size_t i = 0; i < tarr.size() - 1; ++i) {
            ss << (tarr[i] / repeat) << " ";
        }
        ss << (tarr[tarr.size()-1] / repeat) << "]";
        return ss.str();
    }

    void setNumSignificantDigits(int n) { 
        if (n < 1) {
            std::cerr << "Timer: number of significant digits cannot be < 1. Set to 1\n";
            n_sig_dig = 1;
            return;
        }
        n_sig_dig = n;
    }
    
private:
    void calcStats() {
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
};

/// Total running time will be approximately
/// warmupTime + replication * runTime
class Timer {
double warmupTime = 0.05;
double runTime = 0.1;
size_t replication;
int verbose;
public:
    Timer(size_t replication=15, int verbose=0) : replication(replication), verbose(verbose) {
        if (replication > 99) {
            std::cerr << "Timer: replication cannot be lager than 99. Set to 99\n";
            replication = 99;
        }
    }

    void setWarmupTime(double t) { warmupTime = t; }
    void setRunTime(double t) { runTime = t; }

    TimingResult 
    timeit(std::function<void()> method, 
           std::function<void()> setup=[](){},
           std::function<void()> teardown=[](){}) 
    {
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

        return {repeat, replication, tarr };
    }
};

} // namespace timeit

#endif // SIMULATION_TIMEIT_H_