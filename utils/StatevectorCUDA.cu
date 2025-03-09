#include "utils/StatevectorCUDA.h"
#include <curand.h>
#include <curand_kernel.h>

using namespace utils;

template<typename ScalarType>
__global__ void initGaussianKernel(
    ScalarType* dArr, size_t size, ScalarType mean, ScalarType stddev) {
  size_t idx = threadIdx.x + (size_t)(blockIdx.x) * blockDim.x;
  if (idx < size) {
    curandState state;
    curand_init(0, idx, 0, &state);
    dArr[idx] = mean + stddev * curand_normal(&state);
  }
}

template<typename ScalarType>
void utils::internal::HelperCUDAKernels<ScalarType>::randomizeStatevector(
    ScalarType* dArr, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  initGaussianKernel<ScalarType><<<numBlocks, blockSize>>>(dArr, size, 0.0, 1.0);
}
  
template<typename ScalarType>
inline __device__ ScalarType warpReduceSumKernel(ScalarType val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

template<typename ScalarType>
__global__ void reduceSquaredKernel(
    const ScalarType* dArr, ScalarType* dResult, size_t size) {
  __shared__ ScalarType shared[1024];

  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  // Perform sum of squares
  ScalarType localSum = 0.0;
  for (size_t i = idx; i < size; i += stride)
    localSum += dArr[i] * dArr[i];

  // Perform warp-level reduction
  localSum = warpReduceSumKernel(localSum);

  // Use shared memory for inter-warp reduction
  if (threadIdx.x % warpSize == 0)
    shared[threadIdx.x / warpSize] = localSum;

  __syncthreads();

  // Perform block-level reduction using first warp
  localSum = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0;
  if (threadIdx.x < warpSize)
    localSum = warpReduceSumKernel(localSum);

  // Accumulate result using atomic operation
  if (threadIdx.x == 0)
    atomicAdd(dResult, localSum);
}

template<typename ScalarType>
void utils::internal::HelperCUDAKernels<ScalarType>::reduceSquared(
    const ScalarType* dArr, ScalarType* dResult, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  reduceSquaredKernel<ScalarType><<<numBlocks, blockSize>>>(dArr, dResult, size);
}

template<typename ScalarType>
__global__ void multiplyByConstantKernel(
    ScalarType* dArr, ScalarType c, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    dArr[idx] *= c;
}

template<typename ScalarType>
void utils::internal::HelperCUDAKernels<ScalarType>::multiplyByConstant(
    ScalarType* dArr, ScalarType c, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  multiplyByConstantKernel<ScalarType><<<numBlocks, blockSize>>>(dArr, c, size);
}

namespace utils::internal {
  template struct HelperCUDAKernels<float>;
  template struct HelperCUDAKernels<double>;
}