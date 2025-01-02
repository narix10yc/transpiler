#include "saot/CostModel.h"
#include "saot/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>
#include <thread>
#include <simulation/JIT.h>
#include <utils/statevector.h>

using namespace saot;
using namespace llvm;

CostResult NaiveCostModel::computeBenefit(
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
    const QuantumGate &lhsGate, const QuantumGate &rhsGate,
    CircuitGraphContext &context) const {
  assert(0 && "Not Implemented");
  return {0.0, nullptr};
}

StandardCostModel::StandardCostModel(PerformanceCache* cache)
  : cache(cache), updateSpeeds() {
  updateSpeeds.reserve(32);

  for (const auto& item : cache->items) {
    auto it = updateSpeeds.begin();
    auto end = updateSpeeds.end();
    while (it != end) {
      if (it->precision != item.precision || it->nThreads != item.nThreads)
        break;
      ++it;
    }
    if (it == end)
      updateSpeeds.emplace_back(item.nThreads, item.precision, 1, item.memUpdateSpeed);
    else {
      it->nData++;
      it->totalMemSpd += item.memUpdateSpeed;
    }
  }
  std::cerr << "StandardCostModel: "
               "A total of " << updateSpeeds.size() << " items found!\n";
}

double StandardCostModel::computeExpectedMemSpd(const QuantumGate& gate) const {
  assert(0 && "Not Implemented");
  return 0.0;
}

CostResult StandardCostModel::computeBenefit(
    const QuantumGate& lhsGate, const QuantumGate& rhsGate,
    CircuitGraphContext& context) const {
  auto cQubits = lhsGate.qubits;
  for (const auto q : rhsGate.qubits) {
    if (std::ranges::find(cQubits, q) == cQubits.end())
      cQubits.push_back(q);
  }

  auto* cGate = context.quantumGatePool.acquire(rhsGate.lmatmul(lhsGate));
  const auto lSpd = computeExpectedMemSpd(lhsGate);
  const auto rSpd = computeExpectedMemSpd(rhsGate);
  const auto cSpd = computeExpectedMemSpd(*cGate);
  double benefit = 2 * cSpd / (lSpd + rSpd) - 1;

  std::cerr << "lSpd = " << lSpd << ", rSpd = " << rSpd
            << ", cSpd = " << cSpd << "; Benefit = " << benefit << "\n";

  return { benefit, cGate };
}

void PerformanceCache::saveToCSV(const std::string& _fileName) const {
  std::string fileName = _fileName;
  auto l = fileName.size();
  if (l < 4 || fileName.substr(l - 4, l) != ".csv")
    fileName += ".csv";

  std::ofstream file(fileName);
  assert(file.is_open());

  file << "nqubits,opCount,precision,irregularity,nThreads,memSpd\n";
  for (const auto&
      [nqubits, opCount, precision,
       irregularity, nThreads, memUpdateSpeed] : items) {
    file << nqubits << "," << opCount << ","
         << precision << "," << nThreads << ","
         << irregularity << ","
         << std::scientific << std::setw(6) << memUpdateSpeed << "\n";
  }
  file.close();
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
    const CPUKernelGenConfig& cpuConfig,
    int nqubits, int nThreads, int comprehensiveness) {
  assert(nqubits >= 8 && nqubits <= 32);
  assert(comprehensiveness >= 1 && comprehensiveness <= 3);

  auto llvmContext = std::make_unique<LLVMContext>();
  auto llvmModule = std::make_unique<Module>("perfCacheModule", *llvmContext);

  std::vector<QuantumGate> gates;
  gates.reserve(3 * nqubits);
  std::vector<std::unique_ptr<KernelInfo>> kernelInfos;
  kernelInfos.reserve(3 * nqubits);

  utils::timedExecute([&]() {
    std::random_device _rd;
    std::mt19937 gen(_rd());
    std::uniform_int_distribution<> distri(0, nqubits - 1);

    // single-qubit gates
    gates.emplace_back(QuantumGate::RandomUnitary(0));
    gates.emplace_back(QuantumGate::RandomUnitary(nqubits - 1));
    if (comprehensiveness > 2) {
      for (int q = 1; q < nqubits - 1; ++q)
        gates.emplace_back(QuantumGate::RandomUnitary(q));
    }

    // two-qubit gates
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1));
    gates.emplace_back(QuantumGate::RandomUnitary(1, 2));
    gates.emplace_back(QuantumGate::RandomUnitary(nqubits - 3, nqubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(nqubits - 2, nqubits - 1));
    if (comprehensiveness > 2) {
      for (int i = 0; i < nqubits; ++i) {
        int a, b;
        a = distri(gen);
        do { b = distri(gen); } while (b == a);
        gates.emplace_back(QuantumGate::RandomUnitary(a, b));
      }
    }

    // three-qubit gates: (6 + 5 * comprehensiveness) gates
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1, 2));
    gates.emplace_back(QuantumGate::RandomUnitary(1, 2, 3));
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1, 3));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 3, nqubits - 2, nqubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 4, nqubits - 3, nqubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 4, nqubits - 3, nqubits - 1));
    for (int i = 0; i < 5 * comprehensiveness; ++i) {
      int a, b, c;
      a = distri(gen);
      do { b = distri(gen); } while (b == a);
      do { c = distri(gen); } while (c == b || c == a);
      gates.emplace_back(QuantumGate::RandomUnitary(a, b, c));
    }

    // four-qubit gates: (8 + 5 * comprehensiveness) gates
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1, 2, 3));
    gates.emplace_back(QuantumGate::RandomUnitary(1, 2, 3, 4));
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1, 3, 4));
    gates.emplace_back(QuantumGate::RandomUnitary(1, 2, 3, 4));

    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 4, nqubits - 3, nqubits - 2, nqubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 5, nqubits - 4, nqubits - 3, nqubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 5, nqubits - 4, nqubits - 2, nqubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nqubits - 5, nqubits - 3, nqubits - 2, nqubits - 1));
    for (int i = 0; i < 5 * comprehensiveness; ++i) {
      int a, b, c, d;
      a = distri(gen);
      do { b = distri(gen); } while (b == a);
      do { c = distri(gen); } while (c == b || c == a);
      do { d = distri(gen); } while (d == c || d == b || d == a);
      gates.emplace_back(QuantumGate::RandomUnitary(a, b, c, d));
}
  }, "Generate gates for experiments");

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
    if (nThreads == 1)
      tr = timer.timeit([&]() {
        const uint64_t nTasks =
          1ULL << (nqubits - kernel->qubits.size() - kernel->simd_s);
        f(sv.data, 0ULL, nTasks, gates[i].gateMatrix.getConstantMatrix()->data());
      });
    else
      tr = timer.timeit([&]() {
        const uint64_t nTasks =
          1ULL << (nqubits - kernel->qubits.size() - kernel->simd_s);
        std::vector<std::thread> threads;
        threads.reserve(nThreads);
        const uint64_t nTasksPerThread = nTasks / nThreads;
        const auto* matrixPtr = gates[i].gateMatrix.getConstantMatrix()->data();
        for (unsigned tIdx = 0; tIdx < nThreads - 1; ++tIdx) {
          threads.emplace_back(f, sv.data,
            nTasksPerThread * tIdx, nTasksPerThread * (tIdx + 1), matrixPtr);
        }
        threads.emplace_back(f, sv.data,
          nTasksPerThread * (nThreads - 1), nTasks, matrixPtr);

        for (auto& t : threads)
          t.join();
      });
    auto memSpd = calculateMemUpdateSpeed(nqubits, kernel->precision, tr.min);
    items.emplace_back(
      kernel->qubits.size(), kernel->opCount, 64, kernel->nLoBits, 1, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel->qubits.begin(), kernel->qubits.size()));
    std::cerr << ": " << memSpd << " GiBps\n";
  }
}

// PerformanceCache::LoadFromCSV helper functions
namespace {

int parseInt(const char*& curPtr, const char* bufferEnd) {
  const auto* beginPtr = curPtr;
  while (curPtr < bufferEnd && *curPtr >= '0' && *curPtr <= '9')
    ++curPtr;
  assert(curPtr == bufferEnd || *curPtr == ',' || *curPtr == '\n');
  return std::stoi(std::string(beginPtr, curPtr));
}

double parseDouble(const char*& curPtr, const char* bufferEnd) {
  const auto* beginPtr = curPtr;
  while (curPtr < bufferEnd &&
         ((*curPtr >= '0' && *curPtr <= '9') ||
           *curPtr == 'e' || *curPtr == 'E' ||
           *curPtr == '.' || *curPtr == '-' || *curPtr == '+'))
    ++curPtr;
  assert(curPtr == bufferEnd || *curPtr == ',' || *curPtr == '\n');
  return std::stod(std::string(beginPtr, curPtr));
}

PerformanceCache::Item parseLine(const char*& curPtr, const char* bufferEnd) {
  PerformanceCache::Item item;

  item.nqubits = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.opCount = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.precision = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.irregularity = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.nThreads = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.memUpdateSpeed = parseDouble(curPtr, bufferEnd);
  assert(*curPtr == '\n' || curPtr == bufferEnd);
  return item;
}

} // anonymous namespace


PerformanceCache PerformanceCache::LoadFromCSV(const std::string& fileName) {
  PerformanceCache cache;

  std::ifstream file(fileName, std::ifstream::binary);
  assert(file.is_open());

  file.seekg(0, file.end);
  const auto bufferLength = file.tellg();
  file.seekg(0, file.beg);

  auto* bufferBegin = new char[bufferLength];
  auto* bufferEnd = bufferBegin + bufferLength;
  file.read(bufferBegin, bufferLength);
  file.close();
  const auto* curPtr = bufferBegin;
  // parse the header
  while (*curPtr != '\n')
    ++curPtr;
  assert(std::string(bufferBegin, curPtr - bufferBegin) ==
        "nqubits,opCount,precision,irregularity,nThreads,memSpd");
  ++curPtr;

  while (curPtr < bufferEnd) {
    cache.items.push_back(parseLine(curPtr, bufferEnd));
    assert(*curPtr == '\n' || curPtr == bufferEnd);
    if (*curPtr == '\n')
      ++curPtr;
  }

  delete[] bufferBegin;
  return cache;
}