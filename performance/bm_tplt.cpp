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


using real_ty = double;
#define K_VALUE 5

unsigned k = K_VALUE;
constexpr unsigned constK = K_VALUE;

int main() {
    Timer timer;
    TimingResult tr;
    real_ty *real, *imag;
    uint64_t nqubits;

    auto setup = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            imag = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
        };
    };

    auto teardown = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(2));
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

    ComplexMatrix2<real_ty> mat {
        {0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189},
        {0,0.08374721744037222,0.3483304829644841,0.3067364145782554}};

    for (uint64_t nqubit = 6; nqubit < 29; nqubit += 2) {
        // QuEST
        tr = timer.timeit(
            [&](){ applySingleQubitQuEST(real, imag, mat, nqubit, k); },
            setup(nqubit),
            teardown
        );
        std::cerr << "QuEST: nqubits = " << nqubit << "\n";
        tr.display();
        f << "quest" // method
          << "," << "gcc-11" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8*sizeof(real_ty) // real_ty
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << "N/A" // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";

        // double loop
        tr = timer.timeit(
            [&](){ applySingleQubit(real, imag, mat, nqubit, k); },
            setup(nqubit),
            teardown
        );
        std::cerr << "double loop: nqubits = " << nqubit << "\n";
        tr.display();
        f << "double-loop" // method
          << "," << "gcc-11" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8*sizeof(real_ty) // real_ty
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << "N/A" // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";

        // template
        tr = timer.timeit(
            [&](){ applySingleQubitTemplate<real_ty, constK>(real, imag, mat, nqubit); },
            setup(nqubit),
            teardown
        );
        std::cerr << "template: nqubits = " << nqubit << "\n";
        tr.display();
        f << "template" // method
          << "," << "gcc-11" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8*sizeof(real_ty) // real_ty
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << "N/A" // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}