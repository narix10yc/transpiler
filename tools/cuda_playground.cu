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

__global__ void reduceKernel0(const float* dArr, float* dResult, int n) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared[tid] = (idx < n) ? dArr[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    dResult[blockIdx.x] = shared[0];
}

float gpuSum(const float* h_input, int n) {
  float *d_input, *d_intermediate, *h_intermediate;
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  cudaMalloc(&d_input, n * sizeof(float));
  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&d_intermediate, gridSize * sizeof(float));
  h_intermediate = new float[gridSize];

  sumReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, n);
  cudaMemcpy(h_intermediate, d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  // Final reduction on host
  float sum = 0;
  for (int i = 0; i < gridSize; ++i)
      sum += h_intermediate[i];

  cudaFree(d_input);
  cudaFree(d_intermediate);
  delete[] h_intermediate;

  return sum;
}

int main() {
  float* dData;
  float* dResult;
  size_t size = 1ULL << 30;
  int nThreadsPerBlock = 256;
  int nBlocks = (size + nThreadsPerBlock - 1) / nThreadsPerBlock;
  int sharedMemSize = nThreadsPerBlock * sizeof(float);

  cudaMalloc(&dData, size * sizeof(float));
  cudaMalloc(&dResult, sizeof(float));
  fillArrayKernel<float><<<nBlocks, nThreadsPerBlock>>>(dData, 1.0f, size);

  reduceKernel0<<<nBlocks, nThreadsPerBlock, sharedMemSize>>>(dData, dResult, size);
  cudaDeviceSynchronize();

  float hResult;
  cudaMemcpy(&hResult, dResult, sizeof(float), cudaMemcpyDeviceToHost);
  std::cerr << "size: " << size << ", result: " << hResult << "\n";

  timeit::Timer timer;
  auto tr = timer.timeit([&]() {
    reduceKernel0<<<nBlocks, nThreadsPerBlock, sharedMemSize>>>(dData, dResult, size);
    cudaDeviceSynchronize();
  });
  tr.display();


  cudaFree(dResult);
  cudaFree(dData);

  return 0;
}