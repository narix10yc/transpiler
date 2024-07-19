#include "gen_file.h"
#include "utils/iocolor.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"
#include "simulation/tplt.h"
#include <iomanip>

#ifdef USING_F32
    using real_t = float;
#else 
    using real_t = double;
#endif
using Statevector = utils::statevector::StatevectorSep<real_t>;

#define Q0_VALUE 3
#define S_VALUE 3

using namespace Color;
using namespace timeit;
using namespace simulation::tplt;

int main(int argc, char** argv) {
    Timer timer;
    timer.setReplication(15);
    TimingResult rst;

    const std::vector<int> nqubitsVec
        { 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 };
        // {24};

    Statevector sv(nqubitsVec.back());
    std::vector<double> tVec(nqubitsVec.size());

    std::cerr << YELLOW_FG << "Please check: q0 = " << Q0_VALUE << ", s = " << S_VALUE << "\n" << RESET;

    std::cerr << "=== IR ===\n";
    for (unsigned i = 0; i < nqubitsVec.size(); i++) {
        const int nqubits = nqubitsVec[i];
        std::cerr << "nqubits = " << nqubits << "\n";

        if (nqubits >= 30)
            timer.setReplication(5);

        const uint64_t idxMax = 1ULL << (nqubits - S_VALUE - 1);
        rst = timer.timeit(
            [&]() {
                kernel_block_0(sv.real, sv.imag, 0, idxMax, _mPtr);
            }
        );
        // rst.display();
        tVec[i] = rst.min;
    }

    for (const auto& t : tVec) {
        std::cerr << std::scientific << std::setprecision(4) << t << ",";
    }
    std::cerr << "\n";


    std::cerr << "=== QuEST ===\n";
    for (unsigned i = 0; i < nqubitsVec.size(); i++) {
        const int nqubits = nqubitsVec[i];
        std::cerr << "nqubits = " << nqubits << "\n";

        if (nqubits >= 30)
            timer.setReplication(5);

        const uint64_t idxMax = 1ULL << (nqubits - 4);
        rst = timer.timeit(
            [&]() {
                applySingleQubitQuEST<real_t>(sv.real, sv.imag, _mPtr, nqubits, Q0_VALUE);
            }
        );
        // rst.display();
        tVec[i] = rst.min;
    }
    for (const auto& t : tVec) {
        std::cerr << std::scientific << std::setprecision(4) << t << ",";
    }
    std::cerr << "\n";


    std::cerr << "=== Double-loop ===\n";
    for (unsigned i = 0; i < nqubitsVec.size(); i++) {
        const int nqubits = nqubitsVec[i];
        std::cerr << "nqubits = " << nqubits << "\n";

        if (nqubits >= 30)
            timer.setReplication(5);

        const uint64_t idxMax = 1ULL << (nqubits - 4);
        rst = timer.timeit(
            [&]() {
                applySingleQubit<real_t>(sv.real, sv.imag, _mPtr, nqubits, Q0_VALUE);
            }
        );
        // rst.display();
        tVec[i] = rst.min;
    }
    for (const auto& t : tVec) {
        std::cerr << std::scientific << std::setprecision(4) << t << ",";
    }
    std::cerr << "\n";


    std::cerr << "=== Template ===\n";
    for (unsigned i = 0; i < nqubitsVec.size(); i++) {
        const int nqubits = nqubitsVec[i];
        std::cerr << "nqubits = " << nqubits << "\n";

        if (nqubits >= 30)
            timer.setReplication(5);

        const uint64_t idxMax = 1ULL << (nqubits - 4);
        rst = timer.timeit(
            [&]() {
                applySingleQubitTemplate<real_t, Q0_VALUE>(sv.real, sv.imag, _mPtr, nqubits);
            }
        );
        // rst.display();
        tVec[i] = rst.min;
    }
    for (const auto& t : tVec) {
        std::cerr << std::scientific << std::setprecision(4) << t << ",";
    }
    std::cerr << "\n";
    return 0;
}