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

// dArr will be a read-only array with size gridSize * blockSize
// dOut will be a write-only array with size gridSize
// The effect of this function is to write
// dOut[bid] = dArr[bid * blockSize] + 
//             dArr[bid * blockSize + 1] +
//             ... +
//             dArr[bid * blockSize + (blockSize - 1)] +
__global__ void reduceKernel0(const float* dArr, float* dOut) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  size_t bid = blockIdx.x;

  // copy to shared memory
  shared[tid] = dArr[bid * blockDim.x + tid];
  __syncthreads();

  // Reduction in shared memory. The loop here sums the blockSize elements in 
  // \c shared into shared[0].
  for (size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    dOut[bid] = shared[0];
}

__global__ void reduceKernel1(const float* dArr, float* dOut) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  size_t bid = blockIdx.x;
  int blockSize = blockDim.x;

  // copy to shared memory
  shared[tid] = dArr[bid * blockSize + tid];
  __syncthreads();

  // Reduction in shared memory. The loop here sums the blockSize elements in 
  // \c shared into shared[0].
  // Loop unroll
  for (size_t stride = blockSize >> 1; stride > 32; stride >>= 1) {
    if (tid < stride)
      shared[tid] = shared[tid] + shared[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    // float val = shared[tid];
    // val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    // val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    // val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    // val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    // val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    // if (tid == 0)
    //   shared[0] = val;
    volatile float* vshared = shared;
    vshared[tid] = vshared[tid] + vshared[tid + 32];
    vshared[tid] = vshared[tid] + vshared[tid + 16];
    vshared[tid] = vshared[tid] + vshared[tid + 8];
    vshared[tid] = vshared[tid] + vshared[tid + 4];
    vshared[tid] = vshared[tid] + vshared[tid + 2];
    vshared[tid] = vshared[tid] + vshared[tid + 1];
  }
  __syncthreads();
  if (tid == 0)
    dOut[bid] = shared[0];
}


#define kernel reduceKernel1

int main() {
  float* dArr;
  constexpr int blockSize = 512;
  constexpr size_t size = 1ULL << 22;
  constexpr size_t gridSize = (size + blockSize - 1) / blockSize;

  cudaMalloc(&dArr, size * sizeof(float));
  fillArrayKernel<float><<<gridSize, blockSize>>>(dArr, 1.0f, size);
  // writeIncrementalArrayKernel<float><<<gridSize, blockSize>>>(dArr, size);

  float* dIntermediate;
  float* hIntermediate; 
  hIntermediate = new float[gridSize];
  
  cudaMalloc(&dIntermediate, gridSize * sizeof(float));

  size_t sharedMemSize = gridSize * sizeof(float);
  // maintain a reasonable shared memory size

  kernel<<<gridSize, blockSize, sharedMemSize>>>(dArr, dIntermediate);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaMemcpy(hIntermediate, dIntermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  // Final reduction on host
  float sum = 0;
  for (size_t i = 0; i < gridSize; ++i) {
    // std::cerr << "i = " << i << ", hIntermediate[i] = " << hIntermediate[i] << "\n";
    sum += hIntermediate[i];
  }

  std::cerr << "size: " << size << ", sum = " << static_cast<int>(sum) << "\n";

  timeit::Timer timer;
  auto tr = timer.timeit([&]() {  
    kernel<<<gridSize, blockSize, sharedMemSize>>>(dArr, dIntermediate);
    cudaDeviceSynchronize();
  });
  tr.display();

  auto bandwidth = static_cast<double>(size * sizeof(float)) * 1e-9 / tr.min;
  std::cerr << "Bandwidth " << bandwidth << " GiBps\n";

  cudaFree(dArr);

  cudaFree(dIntermediate);
  delete[] hIntermediate;
  return 0;
}