#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include "timeit/timeit.h"

template<typename T>
__global__ void fillArrayKernel(T* dArr, T c, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    dArr[idx] = c;
}

template<typename T>
__global__ void writeIncrementalArrayKernel(T* dArr, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    dArr[idx] = static_cast<T>(idx);
}

template<typename ScalarType, unsigned blockSize>
__global__ void omittingBitReductionKernel(
    const ScalarType* dArr, ScalarType* dOut, size_t size, int bit) {
  static_assert(blockSize == 32 || blockSize == 64 || blockSize == 128 ||
                blockSize == 256 || blockSize == 512);
  __shared__ ScalarType shared[blockSize];
  unsigned tid = threadIdx.x;
  unsigned bid = blockIdx.x;

  size_t i0 = (2ULL * bid) * blockSize + tid;
  size_t i1 = i0 + blockSize;
  // insert a zero into bit-position 'bit'
  size_t mask = (1ULL << bit) - 1;
  i0 = ((i0 & ~mask) << 1) | (i0 & mask);
  i1 = ((i1 & ~mask) << 1) | (i1 & mask);
  shared[tid] = (i0 < size ? dArr[i0] : 0) + 
                (i1 < size ? dArr[i1] : 0);
  __syncthreads();
  if (tid == 0) {
    for (unsigned i = 0; i < blockSize; ++i) {
      printf("%f ", shared[i]);
    }
    printf("\n");
  }
  __syncthreads();

  // Reduction in shared memory. The loop here sums the blockSize elements in 
  // \c shared into shared[0].
  if constexpr (blockSize >= 512) {
    if (tid < 256)
      shared[tid] = shared[tid] + shared[tid + 256];
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128)
      shared[tid] = shared[tid] + shared[tid + 128];
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64)
      shared[tid] = shared[tid] + shared[tid + 64];
    __syncthreads();
  }
  // threads in the same warp. No need to sync
  if (tid < 32) {
    volatile ScalarType* vshared = shared;
    vshared[tid] = vshared[tid] + vshared[tid + 32];
    vshared[tid] = vshared[tid] + vshared[tid + 16];
    vshared[tid] = vshared[tid] + vshared[tid + 8];
    vshared[tid] = vshared[tid] + vshared[tid + 4];
    vshared[tid] = vshared[tid] + vshared[tid + 2];
    vshared[tid] = vshared[tid] + vshared[tid + 1];
  }
  if (tid == 0)
    dOut[bid] = shared[0];
}


int main() {
  float* dArr;
  constexpr int blockSize = 32;
  constexpr size_t size = 1ULL << 9;
  constexpr size_t gridSize = (size + 2 * blockSize - 1) / (2 * blockSize);

  cudaMalloc(&dArr, size * sizeof(float));
  // fillArrayKernel<float><<<size / blockSize, blockSize>>>(dArr, 1.0f, size);
  // writeIncrementalArrayKernel<float><<<size / blockSize, blockSize>>>(dArr, size);

  fillArrayKernel<<<1, 1>>>(dArr, 1.0f, 1);
  fillArrayKernel<<<size / blockSize, blockSize>>>(dArr + 1, 0.0f, size - 1);

  float* dIntermediate;
  float* hIntermediate; 
  hIntermediate = new float[gridSize];
  
  cudaMalloc(&dIntermediate, gridSize * sizeof(float));

  omittingBitReductionKernel<float, blockSize>
    <<<gridSize, blockSize>>>(dArr, dIntermediate, size, 1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaMemcpy(hIntermediate, dIntermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  // Final reduction on host
  float sum = 0;
  for (size_t i = 0; i < gridSize; ++i) {
    std::cerr << "i = " << i << ", hIntermediate[i] = " << hIntermediate[i] << "\n";
    sum += hIntermediate[i];
  }

  std::cerr << "size: " << size << ", sum = " << static_cast<int>(sum) << "\n";

  // timeit::Timer timer;
  // auto tr = timer.timeit([&]() {  
  //   kernel<<<gridSize, blockSize, sharedMemSize>>>(dArr, dIntermediate);
  //   cudaDeviceSynchronize();
  // });
  // tr.display();

  // auto bandwidth = static_cast<double>(size * sizeof(float)) * 1e-9 / tr.min;
  // std::cerr << "Bandwidth " << bandwidth << " GiBps\n";

  cudaFree(dArr);

  cudaFree(dIntermediate);
  delete[] hIntermediate;
  return 0;
}