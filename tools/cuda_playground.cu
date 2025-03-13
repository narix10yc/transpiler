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

// dArr will be a read-only array with size gridSize * blockSize
// dOut will be a write-only array with size gridSize
__global__ void reduceKernel0(const float* dArr, float* dResult) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // copy to shared memory
  shared[tid] = dArr[idx];
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    dResult[blockIdx.x] = shared[0];
}

__inline__ __device__ float warpReduceSumKernel(float val) {
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  return val;
}

__global__ void reduceKernel1(const float* g_idata, float* g_odata) {
  __shared__ int sdata[32];  // Store partial sums

  int tid = threadIdx.x;
  int lane = tid % 32;
  int warpID = tid / 32;

  int val = g_idata[blockIdx.x * blockDim.x + tid];

  // Intra-warp reduction using __shfl_down_sync
  val = warpReduceSumKernel(val);

  // Store per-warp sum in shared memory
  if (lane == 0) sdata[warpID] = val;
  __syncthreads();

  // Final reduction by first warp
  if (warpID == 0) {
      val = (lane < blockDim.x / 32) ? sdata[lane] : 0;
      val = warpReduceSumKernel(val);
      if (lane == 0) g_odata[blockIdx.x] = val;
  }
}

template <unsigned int blockSize>
__device__ void warpReduce(float* sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduceKernel6(const float* g_idata, float* g_odata, size_t n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;
  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32)
    warpReduce(sdata, tid);
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template<size_t size>
float gpuSum(const float* dArr) {
  float *d_intermediate, *h_intermediate;
  constexpr int blockSize = 256;
  constexpr size_t gridSize = (size + blockSize - 1) / blockSize;

  cudaMalloc(&d_intermediate, gridSize * sizeof(float));
  h_intermediate = new float[gridSize];

  reduceKernel6<blockSize><<<gridSize, blockSize, blockSize * sizeof(float)>>>(dArr, d_intermediate, size);
  cudaMemcpy(h_intermediate, d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  // Final reduction on host
  float sum = 0;
  for (size_t i = 0; i < gridSize; ++i) {
    // std::cerr << "i = " << i << ", h_intermediate[i] = " << h_intermediate[i] << "\n";
    sum += h_intermediate[i];
  }

  cudaFree(d_intermediate);
  delete[] h_intermediate;

  return sum;
}

int main() {
  float* dArr;
  constexpr size_t size = 1ULL << 30;
  constexpr int nThreadsPerBlock = 256;
  constexpr int nBlocks = (size + nThreadsPerBlock - 1) / nThreadsPerBlock;

  cudaMalloc(&dArr, size * sizeof(float));
  fillArrayKernel<float><<<nBlocks, nThreadsPerBlock>>>(dArr, 1.0f, size);

  auto sum = gpuSum<size>(dArr);
  std::cerr << "size: " << size << ", sum = " << sum << "\n";

  timeit::Timer timer;
  auto tr = timer.timeit([&]() {
    gpuSum<size>(dArr);
  });
  tr.display();

  auto bandwidth = static_cast<double>(size * sizeof(float)) * 1e-9 / tr.min;
  std::cerr << "Bandwidth " << bandwidth << " GiBps\n";

  cudaFree(dArr);

  return 0;
}