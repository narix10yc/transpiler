#include "timeit/timeit.h"
#include <functional>

using namespace timeit;

int main() {

    Timer timer;
    double *real, *imag;

    auto setup = [&]() {
        real = (double*) malloc(sizeof(double) * (1<<12));
        imag = (double*) malloc(sizeof(double) * (1<<12));
    };

    auto teardown = [&]() {
        free(real); free(imag);
    };

    auto tr = timer.timeit(
        [&]() {
            simulate_circuit(real, imag);
        },
        setup,
        teardown
    );

    tr.display();
    
    return 0;
}