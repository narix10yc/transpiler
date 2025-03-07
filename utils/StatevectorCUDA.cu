#include "utils/StatevectorCUDA.h"

#include <cuda_runtime.h>

#define CALL_CU(FUNC, MSG) \
  cuResult = FUNC; \
  if (cuResult != CUDA_SUCCESS) { \
    std::cerr << RED("[CUDA Err] ") << MSG << ". Error code " \
              << cuResult << "\n"; \
    return; \
  }

#define CALL_CUDA(FUNC, MSG) \
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
  CALL_CUDA(cudaMemset(dData, 0, sizeInBytes()),
    "Failed to zero statevector on the device");
  ScalarType one = 1.0;
  CALL_CUDA(cudaMemcpy(dData, &one, sizeof(ScalarType), cudaMemcpyHostToDevice),
    "Failed to set the first element of the statevector to 1");
  syncState = DeviceIsNewer;
}

template class StatevectorCUDA<float>;
template class StatevectorCUDA<double>;
