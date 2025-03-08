#ifndef UTILS_STATEVECTOR_CUDA_H
#define UTILS_STATEVECTOR_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring> // for std::memcpy
#include <cassert>

namespace utils {

  namespace internal {
    template<typename ScalarType>
    struct HelperCUDAKernels {
      static void randomizeStatevectorCUDA(ScalarType* dData, size_t size);
    };

    extern template struct HelperCUDAKernels<float>;
    extern template struct HelperCUDAKernels<double>;
  } // namespace internal

template<typename ScalarType>
class StatevectorCUDA {
public:
  int nQubits;
  // device data
  ScalarType* dData;
  // host data
  ScalarType* hData;

private:
  enum SyncState {
    UnInited, Synced, DeviceIsNewer, HostIsNewer
  };
  SyncState syncState;
  // cuResult is for CUDA Driver API calls
  CUresult cuResult;
  // cudaResult is for CUDA Runtime API calls
  cudaError_t cudaResult;

  // Malloc device data using CUDA APIs. Users should always check \c dData is
  // null before calling this function.
  void mallocDeviceData();
  void mallocHostData() {
    assert(hData == nullptr && "Host data is already allocated.");  
    hData = (ScalarType*)std::malloc(sizeInBytes());
  }

  void freeDeviceData();
  void freeHostData() { std::free(hData);}
public:
  StatevectorCUDA(int nQubits)
  : nQubits(nQubits), dData(nullptr), hData(nullptr)
  , syncState(UnInited)
  , cuResult(CUDA_SUCCESS), cudaResult(cudaSuccess) {}

  ~StatevectorCUDA() = default;

  StatevectorCUDA(const StatevectorCUDA&);
  StatevectorCUDA(StatevectorCUDA&&);
  StatevectorCUDA& operator=(const StatevectorCUDA&);
  StatevectorCUDA& operator=(StatevectorCUDA&&);

  size_t sizeInBytes() const {
    return (2ULL << nQubits) * sizeof(ScalarType);
  }

  size_t size() const { return 2ULL << nQubits; }

  void initialize();
  
  void randomize() {
    if (dData == nullptr)
      mallocDeviceData();
    internal::HelperCUDAKernels<ScalarType>::randomizeStatevectorCUDA(dData, size());
  }

  void sync();
};

extern template class StatevectorCUDA<float>;
extern template class StatevectorCUDA<double>;
} // namespace utils


#endif // UTILS_STATEVECTOR_CUDA_H
