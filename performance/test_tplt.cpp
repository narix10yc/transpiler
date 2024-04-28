#include "simulation/tplt.h"
#include "timeit/timeit.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <ctime>
#include <thread>
#include <fstream>

std::string getCurrentTime() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d-%H%M%S");
    return ss.str();
}

std::string getOutputFilename() {
    return "out_" + getCurrentTime() + ".csv";
}

using namespace timeit;
using namespace simulation;
using namespace simulation::tplt;

int main() {
    Timer timer;
    TimingResult tr;
    double *real, *imag;
    uint64_t nqubits;

    auto setup = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (double*) aligned_alloc(64, sizeof(double) * (1<<nqubits));
            imag = (double*) aligned_alloc(64, sizeof(double) * (1<<nqubits));
        };
    };

    auto teardown = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(3));
    };

    // open output file
    auto filename = getOutputFilename();
    std::ofstream f { filename };
    if (!f.is_open()) {
        std::cerr << "failed to open " << filename << "\n";
        return 1;
    }

    f << "method,compiler,test_name,real_ty,num_threads,nqubits,k,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    size_t k = 2;
    ComplexMatrix2<double> mat {{0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189},
                {0,0.08374721744037222,0.3483304829644841,0.3067364145782554}};

    for (uint64_t nqubit = 4; nqubit < 28; nqubit += 2) {
        tr = timer.timeit(
            [&](){ applySingleQubitQuEST(real, imag, mat, nqubit, 2); },
            setup(nqubit),
            teardown
        );
        std::cerr << "nqubits = " << nqubit << "\n";
        tr.display();
        f << "quest,clang-17,u3,f64,1," << nqubit << ",2,2,"
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}