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
    Statevector sv(24);

    Timer timer;
    timer.setReplication(5);
    TimingResult rst;

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    std::cerr << "Multi-threading enabled.\n";
    
    // const std::vector<int> nthreadsVec {2,4,8,12,16,20,24,28,32,36};
    // const std::vector<int> nthreadsVec {16,24,32,36,48,64,68,72};
    // const std::vector<int> nthreadsVec {16,20,24,28,32,34,36};

    const std::vector<int> nthreadsVec {2,4};

    for (unsigned i = 0; i < nthreadsVec.size(); i++) {
        int nthreads = nthreadsVec[i];
        std::cerr << nthreads << "-thread:\n";
        uint64_t idxMax;
        uint64_t chunkSize;
        std::vector<std::thread> threads(nthreads);
        for (const auto& data : _metaData) {
            std::cerr << "OpCount " << data.opCount << "; nqubits " << data.nqubits << ": time = ";
            rst = timer.timeit(
                [&]() {
                        idxMax = 1ULL << (sv.nqubits - data.nqubits - S_VALUE);
                        chunkSize = idxMax / nthreads;
                        for (unsigned i = 0; i < nthreads; i++)
                            threads[i] = std::thread(data.func, sv.real, sv.imag, i*chunkSize, (i+1)*chunkSize, data.mPtr);
                        for (unsigned i = 0; i < nthreads; i++)
                            threads[i].join();
                    });
            std::cerr << rst.timeToString(rst.min, 4) << "\n";
        }

    }

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