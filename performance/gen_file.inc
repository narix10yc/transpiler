#include <stdint.h>

typedef double v8double __attribute__((vector_size(64)));

extern "C" {
void u3_0_02003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_1_02001080(double*, double*, uint64_t, uint64_t, v8double);
void u3_2_02003fc0(double*, double*, uint64_t, uint64_t, v8double);
}