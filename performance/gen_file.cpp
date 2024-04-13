#include "../performance/gen_file.inc"

void simulate_circuit(double* real, double* imag) {
  u3_0_a0020c3(real, imag, 0, (1<<8),
    (v8double){-1,0,0,0.522687228930659,0,0,0,-0.8525245220595059});
}

#include "timeit/timeit.h"
#include <functional>

using namespace timeit;

int main() {

    Timer timer;
    double *real, *imag;

    auto setup = [&]() {
        real = (double*) malloc(sizeof(double) * (1<<10));
        imag = (double*) malloc(sizeof(double) * (1<<10));
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