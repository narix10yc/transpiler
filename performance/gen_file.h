#include <stdint.h>

typedef double v8double __attribute__((vector_size(64)));

void u3_0_02003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_1_02001080(double*, double*, uint64_t, uint64_t, v8double);
void u3_2_02003fc0(double*, double*, uint64_t, uint64_t, v8double);
void simulate_circuit(double* real, double* imag) {
  u3_0_02003fff(real, imag, 0, 1,
    (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
  u3_1_02001080(real, imag, 0, 1,
    (v8double){1,0,0,-1,0,0,0,0});
  u3_2_02003fc0(real, imag, 0, 1,
    (v8double){0.9984574954665696,-0.05552143501950479,0.05552143501950479,0.9984574954665696,0,0,0,0});
}