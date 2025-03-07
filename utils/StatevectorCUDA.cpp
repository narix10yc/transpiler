#include "utils/StatevectorCUDA.h"
#include "utils/iocolor.h"

#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

#define CU_CALL(FUNC, MSG) \
  cuResult = FUNC; \
  if (cuResult != CUDA_SUCCESS) { \
    std::cerr << RED("[CUDA Err] ") << MSG << ". Error code " \
              << cuResult << "\n"; \
    return; \
  }

#define CUDA_CALL(FUNC, MSG) \
  cudaResult = FUNC; \
  if (cudaResult != cudaSuccess) { \
    std::cerr << RED("[CUDA Err] ") << MSG << ". " \
              << cudaGetErrorString(cudaResult) << "\n"; \
    return; \
  }

using namespace utils;

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::mallocDeviceData() {
  assert(dData == nullptr && "Already allocated");
  CUDA_CALL(cudaMalloc(&dData, sizeInBytes()),
    "Failed to allocate memory for statevector on the device");
  syncState = DeviceIsNewer;
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::freeDeviceData() {
  CUDA_CALL(cudaFree(dData),
    "Failed to free memory for statevector on the device");
  syncState = HostIsNewer;
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::initialize() {
  if (dData == nullptr)
    mallocDeviceData();
  CUDA_CALL(cudaMemset(dData, 0, sizeInBytes()),
    "Failed to zero statevector on the device");
  ScalarType one = 1.0;
  CUDA_CALL(cudaMemcpy(dData, &one, sizeof(ScalarType), cudaMemcpyHostToDevice),
    "Failed to set the first element of the statevector to 1");
  syncState = DeviceIsNewer;
}

namespace utils {

template class StatevectorCUDA<float>;
template class StatevectorCUDA<double>;

}
