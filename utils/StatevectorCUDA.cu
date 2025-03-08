#include "utils/StatevectorCUDA.h"
#include <curand.h>
#include <curand_kernel.h>

using namespace utils;

template<typename ScalarType>
__global__ void cudaInitGaussian(
    ScalarType* dArr, size_t size, ScalarType mean, ScalarType stddev) {
  size_t idx = threadIdx.x + (size_t)(blockIdx.x) * blockDim.x;
  if (idx < size) {
    curandState state;
    curand_init(0, idx, 0, &state);
    dArr[idx] = mean + stddev * curand_normal(&state);
  }
}

template<typename ScalarType>
void utils::internal::HelperCUDAKernels<ScalarType>::randomizeStatevectorCUDA(
    ScalarType* dData, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  cudaInitGaussian<ScalarType><<<numBlocks, blockSize>>>(dData, size, 0.0, 1.0);
}

namespace utils::internal {
  template struct HelperCUDAKernels<float>;
  template struct HelperCUDAKernels<double>;
}

  
template<typename ScalarType>
inline __device__ ScalarType cudaWarpReduceSum(ScalarType val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

template<typename ScalarType>
__global__ void cudaSumOfSquared(
    const ScalarType* d_in, ScalarType* d_out, size_t size) {
  __shared__ ScalarType shared[1024];

  // Calculate global index
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  // Perform sum of squares
  ScalarType local_sum = 0.0;
  for (size_t i = idx; i < size; i += stride) {
    ScalarType val = d_in[i];
    local_sum += val * val;
  }

  // Perform warp-level reduction
  local_sum = cudaWarpReduceSum(local_sum);

  // Use shared memory for inter-warp reduction
  if (threadIdx.x % warpSize == 0)
    shared[threadIdx.x / warpSize] = local_sum;

  __syncthreads();

  // Perform block-level reduction using first warp
  local_sum = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0;
  if (threadIdx.x < warpSize)
    local_sum = cudaWarpReduceSum(local_sum);

  // Accumulate result using atomic operation
  if (threadIdx.x == 0)
    atomicAdd(d_out, local_sum);
}