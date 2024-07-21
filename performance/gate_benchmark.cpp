#include "gen_file.h"
#include "utils/statevector.h"
#include "utils/iocolor.h"
#include "timeit/timeit.h"
#include <iomanip>
#include <fstream>

#ifdef USING_F32
    using Statevector = utils::statevector::StatevectorSep<float>;
#else 
    using Statevector = utils::statevector::StatevectorSep<double>;
#endif

using namespace timeit;
using namespace Color;

int main(int argc, char** argv) {
    Statevector sv(30);

    Timer timer;
    timer.setReplication(15);
    TimingResult rst;

    if (argc <= 1) {
        std::cerr << RED_FG << "Error: " << RESET << "Need to provide output file\n";
        return 1;
    }
    std::cerr << "-- Output file: " << argv[1] << "\n";

    std::ofstream file(argv[1]);

    #ifdef MULTI_THREAD_SIMULATION_KERNEL
    std::cerr << "Multi-threading enabled.\n";
    
    // const std::vector<int> nthreadsVec {2,4,8,12,16,20,24,28,32,36};
    // const std::vector<int> nthreadsVec {16,24,32,36,48,64,68,72};
    // const std::vector<int> nthreadsVec {16,20,24,28,32,34,36};

    const std::vector<int> nthreadsVec {8, 16};

    uint64_t idxMax;
    uint64_t chunkSize;
    for (const auto& data : _metaData) {
        if ((1ULL << static_cast<int>(std::log2(data.opCount))) != data.opCount)
            continue;
        std::cerr << "OpCount " << data.opCount << "; nqubits " << data.nqubits << "\n";
        file << data.opCount << ",";
        for (unsigned i = 0; i < nthreadsVec.size(); i++) {
            int nthreads = nthreadsVec[i];
            std::vector<std::thread> threads(nthreads);
            rst = timer.timeit(
                [&]() {
                        idxMax = 1ULL << (sv.nqubits - data.nqubits - S_VALUE);
                        chunkSize = idxMax / nthreads;
                        for (unsigned i = 0; i < nthreads; i++)
                            threads[i] = std::thread(data.func, sv.real, sv.imag, i*chunkSize, (i+1)*chunkSize, data.mPtr);
                        for (unsigned i = 0; i < nthreads; i++)
                            threads[i].join();
                    });
            std::cerr << rst.timeToString(rst.min, 4) << " ";
            file << std::scientific << std::setprecision(4) << rst.min;
            if (i < nthreadsVec.size() - 1)
                file << ",";
        }
        file << "\n";
        std::cerr << "\n";

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