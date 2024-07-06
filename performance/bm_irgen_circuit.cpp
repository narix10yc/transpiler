#include "timeit/timeit.h"
#include "gen_file.h"
#include "simulation/statevector.h"

#include <thread>

using namespace timeit;

using real_t = double;
const unsigned nqubits = 28;

using Statevector = simulation::sv::StatevectorSep<real_t>;

int main() {
    Timer timer;
    timer.setReplication(1);
    TimingResult tr;

    Statevector sv(nqubits);
    simulate_circuit(sv.real, sv.imag, nqubits);

    auto setup = [](){};
    auto teardown = [](){};

    tr = timer.timeit(
        [=](){ simulate_circuit(sv.real, sv.imag, nqubits); },
        setup, teardown
    );

    tr.display();

    return 0;
}
