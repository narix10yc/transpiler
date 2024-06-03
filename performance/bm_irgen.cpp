#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

using real_ty = double;
size_t k = 5;
size_t s = 4;
#define SEP_KERNEL f64_s4_sep_u3_k5_33330333
// #define ALT_KERNEL f64_s1_alt_u3_k2_33330333

extern "C" {
    void SEP_KERNEL(real_ty*, real_ty*, uint64_t, uint64_t, void*);
    // void ALT_KERNEL(real_ty*, uint64_t, uint64_t, void*);
}

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

#include "timeit/timeit.h"
#include <functional>

using namespace timeit;

int main() {
    Timer timer;
    TimingResult tr;
    real_ty *real, *imag, *data;
    uint64_t nqubits;

    real_ty m[8] = {0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554};

    auto setup_sep = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            imag = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
        };
    };

    auto teardown_sep = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    };

    // auto setup_alt = [&](uint64_t _nqubits) {
    //     nqubits = _nqubits;
    //     return [&]() {
    //         data = (real_ty*) aligned_alloc(64, 2 * sizeof(real_ty) * (1<<nqubits));
    //     };
    // };

    // auto teardown_alt = [&]() {
    //     free(data);
    //     std::this_thread::sleep_for(std::chrono::seconds(2));
    // };

    // open output file
    auto filename = getOutputFilename();
    std::ofstream f { filename };
    if (!f.is_open()) {
        std::cerr << "failed to open " << filename << "\n";
        return 1;
    }

    f << "method,compiler,test_name,real_ty,num_threads,nqubits,k,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    for (uint64_t nqubit = 6; nqubit < 29; nqubit += 2) {
        // Separate Format
        tr = timer.timeit(
            [&, n=nqubit](){ SEP_KERNEL(real, imag, 0, 1 << (n - s - 1), m); },
            setup_sep(nqubit),
            teardown_sep
        );
        std::cerr << "nqubits = " << nqubit << "\n";
        tr.display();
        f << "ir_gen_sep" // method
          << "," << "clang-17" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8 * sizeof(real_ty) // real_ty
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << s // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";

        // Alternating Format
    //     tr = timer.timeit(
    //         [&, n=nqubit](){ ALT_KERNEL(data, 0, 1 << (n - s), m); },
    //         setup_alt(nqubit),
    //         teardown_alt
    //     );
    //     std::cerr << "nqubits = " << nqubit << "\n";
    //     tr.display();
    //     f << "ir_gen_alt" // method
    //       << "," << "clang-17" // compiler
    //       << "," << "u3" // test_name
    //       << "," << "f" << 8 * sizeof(real_ty) // real_ty
    //       << "," << 1 // num_threads
    //       << "," << nqubit // nqubits
    //       << "," << k // k
    //       << "," << s // s
    //       << ","
    //       << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}