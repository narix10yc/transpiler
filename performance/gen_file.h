#include <cstdint>

extern "C" {
void f64_s1_sep_u2q_k1l0_0000000004104001(double*, double*, uint64_t, uint64_t, const double*);
}

static const double _u3Param[] = {
};

static const double _u2qParam[] = {
 1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

void simulate_circuit(double *real, double *imag, uint64_t, uint64_t, const double*) {
  f64_s1_sep_u2q_k1l0_0000000004104001(real, imag, 0, 9223372036854775808, _u2qParam + 0);
}