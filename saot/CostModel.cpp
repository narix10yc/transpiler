#include "saot/CostModel.h"
#include "saot/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>
#include <simulation/JIT.h>
#include <utils/statevector.h>

using namespace saot;
using namespace llvm;

CostResult StandardCostModel::computeBenefit(
    const QuantumGate& lhsGate, const QuantumGate& rhsGate,
    CircuitGraphContext& context) const {
  auto cQubits = lhsGate.qubits;
  for (const auto q : rhsGate.qubits) {
    if (std::ranges::find(cQubits, q) == cQubits.end())
      cQubits.push_back(q);
  }

  // check fusion eligibility: nqubits
  if (cQubits.size() > this->maxNQubits) {
    // std::cerr << CYAN("Rejected due to maxNQubits\n");
    return { 0.0, nullptr };
  }

  // check fusion eligibility: opCount
  auto* cGate = context.quantumGatePool.acquire(rhsGate.lmatmul(lhsGate));
  if (maxOp > 0 && cGate->opCount(zeroTol) > maxOp) {
    // std::cerr << CYAN("Rejected due to OpCount\n");
    return { 0.0, nullptr };
  }

  // accept candidate
  // std::cerr << GREEN("Fusion accepted!\n");
  return { 1.0, cGate };
}

CostResult AdaptiveCostModel::computeBenefit(
    const QuantumGate& lhsGate, const QuantumGate& rhsGate,
    CircuitGraphContext& context) const {
  assert(0 && "Not Implemented");
  return {0.0, nullptr};
}

void PerformanceCache::saveToCSV(const std::string& _fileName) const {
  std::string fileName = _fileName;
  auto l = fileName.size();
  if (l < 4 || fileName.substr(l - 4, l) != ".csv")
    fileName += ".csv";

  std::ofstream file(fileName);
  assert(file.is_open());

  file << "nqubits,opCount,irregularity,nThreads,memSpd\n";
  for (const auto&
      [nqubits, opCount, irregularity, nThreads, memUpdateSpeed] : items) {
    file << nqubits << "," << opCount << "," << nThreads << ","
         << irregularity << ","
         << std::scientific << std::setw(6) << memUpdateSpeed << "\n";
  }
  file.close();
}

PerformanceCache PerformanceCache::LoadFromCSV(const std::string& fileName) {
  assert(0 && "Not Implemented");
  return PerformanceCache();
}

namespace {

/// @return Speed in Gigabytes per second (GiBps)
double calculateMemUpdateSpeed(int nqubits, int precision, double t) {
  assert(nqubits >= 0 && (precision == 32 || precision == 64));
  assert(t >= 0.0);

  return static_cast<double>((precision == 32 ? 8ULL : 16ULL) << nqubits) * 1e-9 / t;
}

} // anonymous namespace

void PerformanceCache::runExperiments(
    const CPUKernelGenConfig& cpuConfig, int nqubits, int comprehensiveness) {
  auto llvmContext = std::make_unique<LLVMContext>();
  auto llvmModule = std::make_unique<Module>("perfCacheModule", *llvmContext);

  std::vector<QuantumGate> gates;
  gates.reserve(3 * nqubits);
  std::vector<std::unique_ptr<KernelInfo>> kernelInfos;
  kernelInfos.reserve(3 * nqubits);

  utils::timedExecute([&]() {
    for (int q = 0; q < nqubits; ++q) {
      int q1 = (q + 1) % nqubits;
      int q2 = (q + 2) % nqubits;
      gates.emplace_back(QuantumGate::RandomUnitary<1>({q}));
      gates.emplace_back(QuantumGate::RandomUnitary<2>({q, q1}));
      gates.emplace_back(QuantumGate::RandomUnitary<3>({q, q1, q2}));
    }
  }, "Generate random unitary gates");

  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelInfos.emplace_back(genCPUCode(
        *llvmModule, cpuConfig, gate, "gate_" + std::to_string(i++)));
  }, "Code Generation");

  utils::timedExecute([&]() {
    saot::applyLLVMOptimization(*llvmModule, llvm::OptimizationLevel::O1);
  }, "JIT IR Optimization");

  auto jit = createJITSession(std::move(llvmModule), std::move(llvmContext));

  timeit::Timer timer;
  timeit::TimingResult tr;

  utils::StatevectorAlt<double> sv(nqubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (unsigned i = 0, s = kernelInfos.size(); i < s; ++i) {
    const auto& kernel = kernelInfos[i];
    auto f = cantFail(jit->lookup(kernel->llvmFuncName)).toPtr<CPU_FUNC_TYPE>();
    tr = timer.timeit([&]() {
      f(sv.data, 0ULL, 1ULL << (nqubits - kernel->qubits.size() - kernel->simd_s),
        gates[i].gateMatrix.getConstantMatrix()->data());
    });

    auto memSpd = calculateMemUpdateSpeed(nqubits, kernel->precision, tr.min);
    items.emplace_back(
      kernel->qubits.size(), kernel->opCount, kernel->nLoBits, 1, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel->qubits.begin(), kernel->qubits.size()));
    std::cerr << ": " << memSpd << " GiBps\n";
  }


}