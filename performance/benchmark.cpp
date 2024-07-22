#include "gen_file.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"
#include <iomanip>

#ifdef USING_F32
    using Statevector = utils::statevector::StatevectorSep<float>;
#else 
    using Statevector = utils::statevector::StatevectorSep<double>;
#endif
using namespace timeit;

int main(int argc, char** argv) {
    Statevector sv(30);

    Timer timer;
    // timer.setRunTime(1.5);
    timer.setReplication(3);
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
    // for (int nqubits : {26}) {
        if (nqubits < 20)
            timer.setReplication(21);
        else if (nqubits < 26)
            timer.setReplication(11);
        else if (nqubits < 28)
            timer.setReplication(5);
        else
            timer.setReplication(3); 
        uint64_t idxMax = 1ULL << (nqubits - S_VALUE - 3);
        rst = timer.timeit(
            [&]() {
                for (unsigned i = 0; i < nqubits; ++i)
                    _metaData[i].func(sv.real, sv.imag, 0, idxMax, _metaData[i].mPtr);
            }
        );
        std::cerr << "ours,u3," << nqubits << ",f64,"
                  << std::scientific << std::setprecision(4) << (rst.min / nqubits) << "\n";
    }
    #endif

    return 0;
}