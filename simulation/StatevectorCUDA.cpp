#ifdef CAST_USE_CUDA

#include "simulation/StatevectorCUDA.h"
#include "utils/iocolor.h"

#include <cassert>
#include <iostream>

#define CU_CALL(FUNC, MSG) \
  cuResult = FUNC; \
  if (cuResult != CUDA_SUCCESS) { \
    std::cerr << RED("[CUDA Err] ") << MSG << ". Error code " \
              << cuResult << "\n"; \
  }

#define CUDA_CALL(FUNC, MSG) \
  cudaResult = FUNC; \
  if (cudaResult != cudaSuccess) { \
    std::cerr << RED("[CUDA Err] ") << MSG << ". " \
              << cudaGetErrorString(cudaResult) << "\n"; \
  }

using namespace utils;

template<typename ScalarType>
StatevectorCUDA<ScalarType>::StatevectorCUDA(const StatevectorCUDA& other)
: nQubits(other.nQubits), dData(nullptr), hData(nullptr)
, syncState(other.syncState)
, cuResult(CUDA_SUCCESS), cudaResult(cudaSuccess) {
  // copy device data
  if (other.dData != nullptr) {
    mallocDeviceData();
    CUDA_CALL(
      cudaMemcpy(dData, other.dData, sizeInBytes(), cudaMemcpyDeviceToDevice),
      "Failed to copy statevector on the device");
  }
  // copy host data
  if (other.hData != nullptr) {
    mallocHostData();
    std::memcpy(hData, other.hData, sizeInBytes());
  }
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>::StatevectorCUDA(StatevectorCUDA&& other) 
: nQubits(other.nQubits), dData(other.dData), hData(other.hData)
, syncState(other.syncState)
, cuResult(CUDA_SUCCESS), cudaResult(cudaSuccess) {
  other.dData = nullptr;
  other.hData = nullptr;
  other.syncState = UnInited;
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>&
StatevectorCUDA<ScalarType>::operator=(const StatevectorCUDA& other) {
  if (this == &other)
    return *this;
  this->~StatevectorCUDA();
  new (this) StatevectorCUDA(other);
  return *this;
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>&
StatevectorCUDA<ScalarType>::operator=(StatevectorCUDA&& other) {
  if (this == &other)
    return *this;
  this->~StatevectorCUDA();
  new (this) StatevectorCUDA(std::move(other));
  return *this;
}

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

template<typename ScalarType>
ScalarType StatevectorCUDA<ScalarType>::normSquared() const {
  using Helper = utils::internal::HelperCUDAKernels<ScalarType>;
  assert(dData != nullptr && "Device statevector is not initialized");
  return Helper::reduceSquared(dData, size());
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::randomize() {
  using Helper = utils::internal::HelperCUDAKernels<ScalarType>;
  if (dData == nullptr)
    mallocDeviceData();
  Helper::randomizeStatevector(dData, size());

  // normalize the statevector
  auto c = 1.0 / norm();
  Helper::multiplyByConstant(dData, c, size());
  cudaDeviceSynchronize();

  syncState = DeviceIsNewer;
}

namespace utils {
  template class StatevectorCUDA<float>;
  template class StatevectorCUDA<double>;
}

#endif // CAST_USE_CUDA