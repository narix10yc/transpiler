#ifndef UTILS_STATEVECTOR_CUDA_H
#define UTILS_STATEVECTOR_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace utils {

template<typename RealType>
class StatevectorCUDA {
public:
  int nQubits;
  // device data
  RealType* dData;
  // host data
  RealType* hData;

private:
  enum SyncState {
    UnInited, Synced, DeviceIsNewer, HostIsNewer
  };
  SyncState syncState;
  // cuResult is for CUDA Driver API calls
  CUresult cuResult;
  // cudaResult is for CUDA Runtime API calls
  cudaError_t cudaResult;

  void mallocDeviceData();
  void mallocHostData() { hData = (RealType*)std::malloc(sizeInBytes()); }

  void freeDeviceData();
  void freeHostData() { std::free(hData);}
public:
  StatevectorCUDA(int nQubits)
  : nQubits(nQubits), dData(nullptr), hData(nullptr)
  , syncState(UnInited), cuResult(CUDA_SUCCESS) {}

  ~StatevectorCUDA() = default;

  StatevectorCUDA(const StatevectorCUDA&) = delete;
  StatevectorCUDA(StatevectorCUDA&&) = delete;
  StatevectorCUDA& operator=(const StatevectorCUDA&) = delete;
  StatevectorCUDA& operator=(StatevectorCUDA&&) = delete;

  size_t sizeInBytes() const {
    return sizeof(RealType) << (nQubits + 1);
  }

  void initialize();
  
  void randomize();

  void sync();
};

extern template class StatevectorCUDA<float>;
extern template class StatevectorCUDA<double>;
} // namespace utils


#endif // UTILS_STATEVECTOR_CUDA_H
