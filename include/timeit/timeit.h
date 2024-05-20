#ifndef SIMULATION_TIMEIT_H_
#define SIMULATION_TIMEIT_H_

#include <vector>
#include <string>
#include <iostream>
#include <functional>

namespace timeit {

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
    void display(int=4) const;
    std::string raw_string() const;

    void setNumSignificantDigits(int n) { 
        if (n < 1) {
            std::cerr << "Timer: number of significant digits cannot be < 1. Set to 1\n";
            n_sig_dig = 1;
            return;
        }
        n_sig_dig = n;
    }
private:
    void calcStats();
};

/// Total running time will be approximately
/// warmupTime + replication * runTime
class Timer {
double warmupTime = 0.05;
double runTime = 0.1;
unsigned replication;
unsigned verbose;
public:
    Timer(unsigned replication=15, unsigned verbose=0) : replication(replication), verbose(verbose) {
        if (replication > 99) {
            std::cerr << "Timer: replication cannot be lager than 99. Set to 99\n";
            replication = 99;
        }
    }

    void setWarmupTime(double t) { warmupTime = t; }
    void setRunTime(double t) { runTime = t; }
    void setReplication(unsigned r) { replication = r; }

    TimingResult 
    timeit(std::function<void()> method, 
           std::function<void()> setup=[](){},
           std::function<void()> teardown=[](){});

};

} // end namespace timeit

#endif // SIMULATION_TIMEIT_H_