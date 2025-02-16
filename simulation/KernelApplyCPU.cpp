#include "simulation/KernelManager.h"
#include "cast/CircuitGraph.h"

using namespace cast;

void KernelManager::applyCPUKernel(
    void* sv, int nQubits, const std::string& funcName) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName == funcName) {
      applyCPUKernel(sv, nQubits, kernel);
      return;
    }
  }
  llvm_unreachable("KernelManager::applyCPUKernel: kernel not found by name");
}

void KernelManager::applyCPUKernel(
    void* sv, int nQubits, KernelInfo& kernel) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate.nQubits() - kernel.simd_s;
  assert(tmp > 0);
  uint64_t idxEnd = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernel.gate.gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernel.gate.gateMatrix.getConstantMatrix()->data();
  kernel.executable(sv, 0ULL, idxEnd, pMat);
}

void KernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, KernelInfo& kernel, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  ensureExecutable(kernel);
  int tmp = nQubits - kernel.gate.nQubits() - kernel.simd_s;
  assert(tmp > 0);
  uint64_t nTasks = 1ULL << tmp;
  const void* pMat = nullptr;
  if (kernel.gate.gateMatrix.getConstantMatrix() != nullptr)
    pMat = kernel.gate.gateMatrix.getConstantMatrix()->data();

  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  const uint64_t nTasksPerThread = nTasks / nThreads;

  for (unsigned tIdx = 0; tIdx < nThreads - 1; ++tIdx) {
    threads.emplace_back(kernel.executable, sv,
      nTasksPerThread * tIdx, nTasksPerThread * (tIdx + 1), pMat);
  }
  threads.emplace_back(kernel.executable, sv,
    nTasksPerThread * (nThreads - 1), nTasks, pMat);
  for (auto& t : threads)
    t.join();
}

void KernelManager::applyCPUKernelMultithread(
    void* sv, int nQubits, const std::string& funcName, int nThreads) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::applyCPUKernel");
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName == funcName) {
      applyCPUKernelMultithread(sv, nQubits, kernel, nThreads);
      return;
    }
  }
  llvm_unreachable("KernelManager::applyCPUKernelMultithread: "
                   "kernel not found by name");
}


namespace {
  /// mangled name is formed by 'G' + <length of graphName> + graphName
  /// @return mangled name
  std::string mangleGraphName(const std::string& graphName) {
    return "G" + std::to_string(graphName.length()) + graphName;
  }

  /// @return demangled name
  std::string demangleGraphBlockName(const std::string& mangledName) {
    const auto* p = mangledName.data();
    const auto* e = mangledName.data() + mangledName.size();
    assert(p != e);
    assert(*p == 'G' && "Mangled graph name must start with 'G'");
    ++p;
    assert(p != e);
    auto p0 = p;
    while ('0' <= *p && *p <= '9') {
      ++p;
      assert(p != e);
    }
    auto l = std::stoi(std::string(p0, p));
    assert(p + l <= e);
    return std::string(p, p+l);
  }
} // anonymous namespace


KernelManager& KernelManager::genCPUFromGraph(
    const CPUKernelGenConfig& config, const CircuitGraph& graph,
    const std::string& graphName) {
  const auto allBlocks = graph.getAllBlocks();
  const auto mangledName = mangleGraphName(graphName);
  for (const auto& block : allBlocks) {
    genCPUKernel(
      config, *block->quantumGate,mangledName + std::to_string(block->id));
  }
  return *this;
}

std::vector<KernelInfo*> KernelManager::collectCPUGraphKernels(
    const std::string& graphName) {
  assert(isJITed() && "Must initialize JIT session "
                      "before calling KernelManager::collectCPUGraphKernels");
  std::vector<KernelInfo*> kernelInfos;
  const auto mangledName = mangleGraphName(graphName);
  for (auto& kernel : _kernels) {
    if (kernel.llvmFuncName.starts_with(mangledName)) {
      kernel.executable =
          cantFail(llvmJIT->lookup(kernel.llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
      kernelInfos.push_back(&kernel);
    }
  }
  return kernelInfos;
}

