#include "gen_file.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"

#ifdef USING_F32
    using Statevector = utils::statevector::StatevectorSep<float>;
#else 
    using Statevector = utils::statevector::StatevectorSep<double>;
#endif
using namespace timeit;

int main(int argc, char** argv) {
    Statevector sv(30);

    Timer timer;
    timer.setReplication(1);
    TimingResult rst;

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    std::cerr << "Multi-threading enabled.\n";
    
    const std::vector<int> nthreads {36};

    int warmUpNThread = nthreads[nthreads.size()-1];
    std::cerr << "Warm up run (" << warmUpNThread << "-thread):\n";
    rst = timer.timeit(
        [&]() {
            simulation_kernel(sv.real, sv.imag, warmUpNThread);
        }
    );
    rst.display();

    for (const int nthread : nthreads) {
        std::cerr << nthread << "-thread:\n";
        rst = timer.timeit(
            [&]() {
                simulation_kernel(sv.real, sv.imag, nthread);
            }
        );
        rst.display();
    }

    #else
    rst = timer.timeit(
        [&]() {
            simulation_kernel(sv.real, sv.imag);
        }
    );
    rst.display();
    #endif

    return 0;
}