#include "gen_file.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"
#include <iomanip>
#include <iostream>

#ifdef USING_F32
    using Statevector = utils::statevector::StatevectorSep<float>;
    using real_t = float;
    #define REAL_T "f32"
#else 
    using Statevector = utils::statevector::StatevectorSep<double>;
    using real_t = double;
    #define REAL_T "f64"
#endif
using namespace timeit;

int main(int argc, char** argv) {
    real_t *real, *imag;

        real = (real_t*) std::aligned_alloc(64, 2 * (1ULL << 30) * sizeof(real_t));
        imag = real + (1ULL << 30);

    Timer timer;
    timer.setRunTime(1.5);
    // timer.setReplication(3);
    TimingResult rst;

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    std::cerr << "Multi-threading enabled.\n";
    
    // const std::vector<int> nthreads {2,4,8,12,16,20,24,28,32,36};
    // const std::vector<int> nthreads {16,24,32,36,48,64,68,72};
    const std::vector<int> nthreads {4, 8, 16, 24, 32};

    // const std::vector<int> nthreads {32, 64};

    std::vector<double> tarr(nthreads.size());

    int warmUpNThread = nthreads[nthreads.size()-1];
    std::cerr << "Warm up run (" << warmUpNThread << "-thread):\n";
    rst = timer.timeit(
        [&]() {
            simulation_kernel(sv.real, sv.imag, sv.nqubits, warmUpNThread);
        }
    );
    rst.display();

    for (unsigned i = 0; i < nthreads.size(); i++) {
        int nthread = nthreads[i];
        std::cerr << nthread << "-thread:\n";
        rst = timer.timeit(
            [&]() {
                simulation_kernel(sv.real, sv.imag, sv.nqubits, nthread);
            }
        );
        rst.display();
        tarr[i] = rst.min;
    }
    // easy to copy paste
    for (const auto& t : tarr)
        std::cerr << std::fixed << std::setprecision(4) << t << ",";
    std::cerr << "\n";

    #else
    for (int nqubits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}) {
    // for (int nqubits : {16, 18, 20, 22, 24, 26, 28, 30}) {
    // for (int nqubits : {14, 14, 14}) {
        if (nqubits < 20)
            timer.setReplication(7);
        else if (nqubits < 26)
            timer.setReplication(5);
        else if (nqubits < 28)
            timer.setReplication(3);
        else
            timer.setReplication(1); 
        uint64_t idxMax = 1ULL << (nqubits - S_VALUE - _metaData[0].nqubits);
        // std::cerr << "nqubits = " << nqubits << "\n";
        // real = (real_t*) std::aligned_alloc(64, 2 * (1ULL << nqubits) * sizeof(real_t));
        // imag = real + (1ULL << nqubits);

        double t_min = 999999;
        for (unsigned rep = 0; rep < 3; rep++) {
            rst = timer.timeit(
            [&]() {
                for (unsigned i = 0; i < nqubits; ++i) {
                    _metaData[i].func(real, imag, 0, idxMax, _metaData[i].mPtr);
                }
            });
            // rst.display();  
            if (t_min > rst.min)
                t_min = rst.min;
        }
            
        std::cerr << "ours,u2," << nqubits << "," REAL_T ","
                  << std::scientific << std::setprecision(4) << (t_min / nqubits) << "\n";
        
    }
        std::free(real);
    #endif

    return 0;
}