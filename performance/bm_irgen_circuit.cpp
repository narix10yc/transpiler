#include "timeit/timeit.h"
#include "gen_file.h"
#include "simulation/statevector.h"

#include <thread>

using namespace timeit;

using real_ty = double;
const unsigned nqubits = 24;

using Statevector = simulation::sv::StatevectorSep<real_ty>;

int main() {
    Timer timer;
    timer.setReplication(7);
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
