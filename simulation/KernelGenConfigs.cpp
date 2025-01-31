#include "simulation/KernelManager.h"

using namespace saot;

const CPUKernelGenConfig CPUKernelGenConfig::NativeDefaultF32 {
  // #ifdef __AVX512__
  // .simd_s = 4,
  // #else
  // #ifdef __AVX2__
  // .simd_s = 3,
  // #else
  // #ifdef __NEON__
  // .simd_s = 1
  // #endif
};