#include "gen_file.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"

using Statevector = utils::statevector::StatevectorSep<double>;
using namespace timeit;

int main(int argc, char** argv) {
    Statevector sv(34);

    Timer timer;
    timer.setReplication(1);

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    std::cerr << "Multi-threading enabled.\n";
    // for (int nthread : {2,4,6,8,10,12,16,20,24,28,32,36}) {
    for (int nthread : {18, 36}) {
        std::cerr << "nthreads = " << nthread << "\n";
        auto rst = timer.timeit(
            [&]() {
                simulation_kernel(sv.real, sv.imag, nthread);
            }
        );
        rst.display();
    }

    #else
    auto rst = timer.timeit(
        [&]() {
            simulation_kernel(sv.real, sv.imag);
        }
    );
    rst.display();
    #endif

    return 0;
}