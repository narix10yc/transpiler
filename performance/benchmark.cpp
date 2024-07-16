#include "gen_file.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"

using namespace utils::statevector;
using namespace timeit;

int main(int argc, char** argv) {
    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    int nthreads = std::stoi(argv[1]);
    std::cerr << "Multi-threading enabled. Number of threads: " << nthreads << "\n";
    #endif

    StatevectorSep<double> sv(28);

    Timer timer;
    timer.setReplication(1);

    auto rst = timer.timeit(
        [&]() {
            #ifdef MULTI_THREAD_SIMULATION_KERNEL
            simulation_kernel(sv.real, sv.imag, nthreads);
            #else
            simulation_kernel(sv.real, sv.imag);
            #endif
        }
    );
    rst.display();

    return 0;
}