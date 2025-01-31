#include "simulation/KernelManager.h"

using namespace saot;

void KernelManager::applyCPUKernel(
    void* sv, int nQubits, const std::string& funcName) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");

  auto kernelIt = std::ranges::find_if(_kernels,
    [&funcName](const KernelInfo& k) {
      return k.llvmFuncName == funcName;
    });

  assert(kernelIt != _kernels.end());

  auto f = cantFail(llvmJIT->lookup(funcName)).toPtr<CPU_KERNEL_TYPE>();

  int tmp = nQubits - kernelIt->gate.nqubits() - kernelIt->simd_s;
  assert(tmp > 0);
  uint64_t idxEnd = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernelIt->gate.gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernelIt->gate.gateMatrix.getConstantMatrix()->data();
  f(sv, 0ULL, idxEnd, pMat);
}


void KernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, const std::string& funcName, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");

  auto kernelIt = std::ranges::find_if(_kernels,
    [&funcName](const KernelInfo& k) {
      return k.llvmFuncName == funcName;
    });

  assert(kernelIt != _kernels.end());

  auto f = cantFail(llvmJIT->lookup(funcName)).toPtr<CPU_KERNEL_TYPE>();

  int tmp = nQubits - kernelIt->gate.nqubits() - kernelIt->simd_s;
  assert(tmp > 0);
  uint64_t nTasks = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernelIt->gate.gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernelIt->gate.gateMatrix.getConstantMatrix()->data();

  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  const uint64_t nTasksPerThread = nTasks / nThreads;
  for (unsigned tIdx = 0; tIdx < nThreads - 1; ++tIdx) {
    threads.emplace_back(f, sv,
      nTasksPerThread * tIdx, nTasksPerThread * (tIdx + 1), pMat);
  }
  threads.emplace_back(f, sv,
    nTasksPerThread * (nThreads - 1), nTasks, pMat);

  for (auto& t : threads)
    t.join();
}