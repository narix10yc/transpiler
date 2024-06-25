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


using real_t = double;
constexpr size_t k = 1;
constexpr size_t l = 0;

int main() {
    Timer timer;
    TimingResult tr;
    real_t *real, *imag;
    uint64_t nqubits;

    auto setup = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (real_t*) aligned_alloc(64, sizeof(real_t) * (1<<nqubits));
            imag = (real_t*) aligned_alloc(64, sizeof(real_t) * (1<<nqubits));
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

    f << "method,compiler,test_name,real_t,num_threads,nqubits,k,l,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    auto mat = ComplexMatrix4<real_t>::Random();

    for (uint64_t nqubit = 6; nqubit < 28; nqubit += 2) {
        // QuEST
        tr = timer.timeit(
            [&](){ applyTwoQubitQuEST(real, imag, mat, nqubit, k, l); },
            setup(nqubit),
            teardown
        );
        std::cerr << "QuEST: nqubits = " << nqubit << "\n";
        tr.display();
        f << "quest" // method
          << "," << "clang-17" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8*sizeof(real_t) // real_t
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << l // l
          << "," << "N/A" // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";

        // template
        // tr = timer.timeit(
        //     [&](){ applySingleQubitTemplate<real_t, k>(real, imag, mat, nqubit); },
        //     setup(nqubit),
        //     teardown
        // );
        // std::cerr << "template: nqubits = " << nqubit << "\n";
        // tr.display();
        // f << "template" // method
        //   << "," << "clang-17" // compiler
        //   << "," << "u3" // test_name
        //   << "," << "f" << 8*sizeof(real_t) // real_t
        //   << "," << 1 // num_threads
        //   << "," << nqubit // nqubits
        //   << "," << k // k
        //   << "," << "N/A" // s
        //   << ","
        //   << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}