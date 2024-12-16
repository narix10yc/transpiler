#include "saot/CostModel.h"
#include "saot/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>
#include <simulation/JIT.h>
#include <utils/statevector.h>

using namespace saot;
using namespace llvm;

int StandardCostModel::getCost(const QuantumGate& gate) const {
  assert(0 && "Not Implemented");
  return -1;
}

int AdaptiveCostModel::getCost(const QuantumGate& gate) const {
  assert(0 && "Not Implemented");
  return -1;
}

void PerformanceCache::saveToCSV(const std::string& _fileName) const {
  std::string fileName = _fileName;
  auto l = fileName.size();
  if (l >= 4 && fileName.substr(l - 4, l) != ".csv")
    fileName += ".csv";

  std::ofstream file(fileName);
  assert(file.is_open());

  file << "nqubits,op_count,nthreads,time\n";
  for (const auto &[nqubits, opCount, nThreads, memUpdateSpeed] : items) {
    file << nqubits << "," << opCount << "," << nThreads << ","
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

  std::vector<std::pair<QuantumGate, std::string>> u1qGates, u2qGates, u3qGates;
  u1qGates.reserve(nqubits);
  u2qGates.reserve(nqubits);
  u3qGates.reserve(nqubits);

  utils::timedExecute([&]() {
    for (unsigned q = 0; q < nqubits; ++q) {
      unsigned q1 = (q + 1) % nqubits;
      unsigned q2 = (q + 2) % nqubits;
      u1qGates.emplace_back(
        QuantumGate::RandomU1q(q), "cpu_u1q_" + std::to_string(q));
      u2qGates.emplace_back(
        QuantumGate::RandomU2q(q, q1), "cpu_u2q_" + std::to_string(q));
      u3qGates.emplace_back(
        QuantumGate::RandomU3q(q, q1, q2), "cpu_u3q_" + std::to_string(q));
    }
  }, "Generate random unitary gates");

  utils::timedExecute([&]() {
    for (const auto &[gate, name] : u1qGates)
      genCPUCode(*llvmModule, cpuConfig, gate, name);
    for (const auto &[gate, name] : u2qGates)
      genCPUCode(*llvmModule, cpuConfig, gate, name);
    for (const auto &[gate, name] : u3qGates)
      genCPUCode(*llvmModule, cpuConfig, gate, name);
  }, "Code Generation");


  auto jit = createJITSession(std::move(llvmModule), std::move(llvmContext));
  timeit::Timer timer;
  timeit::TimingResult tr;

  utils::StatevectorAlt<double> sv(nqubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (unsigned q = 0; q < nqubits; ++q) {
    auto fU1q = cantFail(jit->lookup(u1qGates[q].second)).toPtr<CPU_FUNC_TYPE>();
    tr = timer.timeit([&]() {
      fU1q(sv.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s),
        u1qGates[q].first.gateMatrix.getConstantMatrix()->data());
    });

    std::cerr << "U1q @ " << q << ": "
              << calculateMemUpdateSpeed(nqubits, 64, tr.min) << " GiBps\n";
  }

  for (unsigned q = 0; q < nqubits; ++q) {
    auto fU2q = cantFail(jit->lookup(u2qGates[q].second)).toPtr<CPU_FUNC_TYPE>();
    tr = timer.timeit([&]() {
      fU2q(sv.data, 0ULL, 1ULL << (nqubits - 2 - cpuConfig.simd_s),
        u1qGates[q].first.gateMatrix.getConstantMatrix()->data());
    });

    std::cerr << "U2q @ [" << q << "," << q+1 << "]: "
              << calculateMemUpdateSpeed(nqubits, 64, tr.min) << " GiBps\n";
  }

  for (unsigned q = 0; q < nqubits; ++q) {
    auto fU3q = cantFail(jit->lookup(u3qGates[q].second)).toPtr<CPU_FUNC_TYPE>();
    tr = timer.timeit([&]() {
      fU3q(sv.data, 0ULL, 1ULL << (nqubits - 3 - cpuConfig.simd_s),
        u1qGates[q].first.gateMatrix.getConstantMatrix()->data());
    });

    std::cerr << "U3q @ [" << q << "," << (q+1) % nqubits << "," << (q+2) % nqubits << "]: "
              << calculateMemUpdateSpeed(nqubits, 64, tr.min) << " GiBps\n";
  }
  // assert(0 && "Not Implemented");

}