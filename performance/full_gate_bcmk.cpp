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
    Timer timer;
    timer.setRunTime(0.5);
    // timer.setReplication(3);
    TimingResult rst;

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    #else
    std::cerr << "\n============ New Run ============\n";
    int count = 0;
    for (int nqubits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}) {
    // for (const int nqubits : { DEFAULT_NQUBITS, DEFAULT_NQUBITS }) {
        if (nqubits < 20)
            timer.setReplication(7);
        else
            timer.setReplication(3); 
        uint64_t idxMax = 1ULL << (nqubits - S_VALUE - _metaData[0].nqubits);

        real = (real_t*) std::aligned_alloc(64, 2 * (1ULL << nqubits) * sizeof(real_t));
        imag = real + (1ULL << nqubits);

        rst = timer.timeit(
        [&]() {
            for (unsigned i = 0; i < nqubits; ++i) {
                #ifdef USING_ALT_KERNEL
                    _metaData[i].func(real, 0, idxMax, _metaData[i].mPtr);
                #else
                    _metaData[count + i].func(real, imag, 0, idxMax, _metaData[count + i].mPtr);
                #endif
            }
        });
        count += nqubits;
        // rst.display();  
        std::free(real);

        std::cerr << "ours,u2," << nqubits << "," REAL_T ","
                  << std::scientific << std::setprecision(4) << (rst.min / nqubits) << "\n";
    }
    #endif

    return 0;
}