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
    Timer timer;
    timer.setReplication(15);
    TimingResult rst;

    const std::vector<int> nqubitsVec
        // { 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34 };
        {24};

    Statevector sv(nqubitsVec.back());

    std::vector<double> tVec(nqubitsVec.size());
    for (unsigned i = 0; i < nqubitsVec.size(); i++) {
        const int nqubits = nqubitsVec[i];
        std::cerr << "nqubits = " << nqubits << "\n";

        if (nqubits >= 30)
            timer.setReplication(5);

        const uint64_t idxMax = 1ULL << (nqubits - 4);
        rst = timer.timeit(
            [&]() {
                kernel_block_0(sv.real, sv.imag, 0, idxMax, _mPtr);
            }
        );
        rst.display();
        tVec[i] = rst.min;
    }

    for (const auto& t : tVec) {
        std::cerr << std::scientific << std::setprecision(4) << t << ",";
    }
    std::cerr << "\n";

    return 0;
}