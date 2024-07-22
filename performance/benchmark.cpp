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
    timer.setReplication(1);
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
    rst = timer.timeit(
        [&]() {
            simulation_kernel(sv.real, sv.imag, sv.nqubits);
        }
    );
    rst.display();
    #endif

    return 0;
}