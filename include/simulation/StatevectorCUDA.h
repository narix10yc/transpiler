#ifndef UTILS_STATEVECTOR_CUDA_H
#define UTILS_STATEVECTOR_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring> // for std::memcpy
#include <cassert>
#include <cmath>

namespace utils {
  namespace internal {
    template<typename ScalarType>
    struct HelperCUDAKernels {
      static void multiplyByConstant(
        ScalarType* dArr, ScalarType constant, size_t size);

      // Randomize the \c dArr array using standard normal distribution
      static void randomizeStatevector(ScalarType* dArr, size_t size);

      // Return the sum of the squared values of \c dArr array
      static ScalarType reduceSquared(
          const ScalarType* dArr, size_t size);

      static ScalarType reduceSquaredOmittingBit(
          const ScalarType* dArr, size_t size, int bit);
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
  mutable CUresult cuResult;
  // cudaResult is for CUDA Runtime API calls
  mutable cudaError_t cudaResult;

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

  // The size of statevector array, equaling to 2ULL << nQubits.
  size_t size() const { return 2ULL << nQubits; }

  void initialize();

  ScalarType normSquared() const;
  ScalarType norm() const { return std::sqrt(normSquared()); }

  ScalarType prob(int qubits) const;

  void randomize();

  // TODO: Not implemented yet. Sync hData and hData based on syncState
  void sync();
};

extern template class StatevectorCUDA<float>;
extern template class StatevectorCUDA<double>;
} // namespace utils


#endif // UTILS_STATEVECTOR_CUDA_H
