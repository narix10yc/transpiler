#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

using real_ty = double;
constexpr size_t k = 3;
constexpr size_t l = 2;

constexpr size_t s = 2;
constexpr unsigned nthreads = 1;
#define KERNEL f64_s2_sep_u2q_k3l2_ffffffffffffffff
// #define KERNEL f64_s1_sep_u2q_k2l1_batched

extern "C" {
    void KERNEL(real_ty*, real_ty*, uint64_t, uint64_t, void*);
    // void KERNEL(real_ty*, real_ty*, real_ty*, real_ty*, uint64_t, uint64_t, void*);
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
#include <random>

using namespace timeit;

int main() {
    Timer timer;
    TimingResult tr;
    real_ty *real, *imag, *real2, *imag2;
    uint64_t nqubits;

    real_ty m[32];
    std::random_device rd;
    std::mt19937 gen { rd() };
    std::normal_distribution<> d { 0, 1 };
    for (size_t i = 0; i < 32; i++)
        m[i] = d(gen);
    
    auto setup_sep = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            imag = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            // real2 = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            // imag2 = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
        };
    };

    auto teardown_sep = [&]() {
        free(real); free(imag);
        // free(real2); free(imag2);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    };

    // open output file
    auto filename = getOutputFilename();
    std::ofstream f { filename };
    if (!f.is_open()) {
        std::cerr << "failed to open " << filename << "\n";
        return 1;
    }

    f << "method,compiler,test_name,real_ty,num_threads,nqubits,k,l,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    for (uint64_t nqubit = 6; nqubit < 29; nqubit += 2) {
        // Separate Format
        std::function<void(void)> task;
        uint64_t chunkSize = (1 << (nqubit - s - 2)) / nthreads;
        std::thread threads[nthreads];
        if (nthreads == 1)
            task = [&](){
                        KERNEL(real, imag, 0, 1 << (nqubit - s - 2), m);
                    };
        else
            task = [&]() {
                for (int i = 0; i < nthreads; i++)
                    threads[i] = std::thread{KERNEL, real, imag, i*chunkSize, (i+1)*chunkSize, m};
                for (auto& t : threads)
                    t.join();
            };
    
        tr = timer.timeit(
            task,
            setup_sep(nqubit),
            teardown_sep
        );
        std::cerr << "nqubits = " << nqubit << "\n";
        tr.display();
        f << "ir_gen_sep" // method
          << "," << "clang-17" // compiler
          << "," << "u2q(ffffffffffffffff)" // test_name
          << "," << "f" << 8 * sizeof(real_ty) // real_ty
          << "," << nthreads // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << l // l
          << "," << s // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}