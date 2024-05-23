#include "timeit/timeit.h"
#include "gen_file.h"

#include <thread>


using namespace timeit;

using real_ty = float;
const unsigned nqubits = 24;

int main() {
    Timer timer;
    TimingResult tr;
    real_ty *real, *imag;
    timer.setReplication(7);

    auto setup = [&]() {
        real = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
        imag = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
    };

    auto teardown = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    };

    tr = timer.timeit(
        [&](){ simulate_circuit(real, imag, nqubits); },
        setup, teardown
    );

    tr.display();
}