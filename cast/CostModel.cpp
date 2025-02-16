#include "cast/CostModel.h"
#include "cast/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>
#include <llvm/IR/InlineAsm.h>

#include "utils/statevector.h"

using namespace cast;
using namespace llvm;

double NaiveCostModel::computeSpeed(
    const QuantumGate& gate, int precision, int nThreads) const {
  if (gate.nQubits() > maxnQubits)
    return 1e-8;
  if (gate.opCount(zeroTol) > maxOp)
    return 1e-8;

  return 1.0;
}

StandardCostModel::StandardCostModel(PerformanceCache* cache, double zeroTol)
  : cache(cache), zeroTol(zeroTol), updateSpeeds() {
  updateSpeeds.reserve(32);

  for (const auto& item : cache->items) {
    auto it = updateSpeeds.begin();
    auto end = updateSpeeds.end();
    while (it != end) {
      if (it->nQubits == item.nQubits && it->precision == item.precision &&
          it->nThreads == item.nThreads)
        break;
      ++it;
    }
    if (updateSpeeds.empty() || it == end)
      updateSpeeds.emplace_back(
        item.nQubits, item.precision, item.nThreads, 1,
        1.0 / (item.opCount * item.memUpdateSpeed));
    else {
      it->nData++;
      it->totalTimePerOpCount += 1.0 / (item.opCount * item.memUpdateSpeed);
    }
  }
  std::cerr << "StandardCostModel: "
               "A total of " << updateSpeeds.size() << " items found!\n";
}

double StandardCostModel::computeSpeed(
    const QuantumGate& gate, int precision, int nThreads) const {
  assert(!updateSpeeds.empty());
  const auto gatenQubits = gate.nQubits();

  // Try to find an exact match
  for (const auto& item : updateSpeeds) {
    if (item.nQubits == gatenQubits && item.precision == precision
        && item.nThreads == nThreads) {
      return item.getMemSpd(gate.opCount(zeroTol));
    }
  }

  // No exact match. Estimate it
  auto bestMatchIt = updateSpeeds.begin();

  auto it = updateSpeeds.cbegin();
  const auto end = updateSpeeds.cend();
  while (++it != end) {
    // priority: nThreads > nQubits > precision
    const int bestDiffNThreads = std::abs(nThreads - bestMatchIt->nThreads);
    const int thisDiffNThreads = std::abs(nThreads - it->nThreads);
    if (thisDiffNThreads > bestDiffNThreads)
      continue;
    if (thisDiffNThreads < bestDiffNThreads) {
      bestMatchIt = it;
      continue;
    }

    const int bestDiffnQubits = std::abs(gatenQubits - bestMatchIt->nQubits);
    const int thisDiffnQubits = std::abs(gatenQubits - it->nQubits);
    if (thisDiffnQubits > bestDiffnQubits)
      continue;
    if (thisDiffnQubits < bestDiffnQubits) {
      bestMatchIt = it;
      continue;
    }

    if (precision == bestMatchIt->precision)
      continue;
    if (precision == it->precision) {
      bestMatchIt = it;
      continue;
    }
  }

  int bestMatchOpCount = 1 << (2 * bestMatchIt->nQubits + 2);
  double memSpd = bestMatchIt->getMemSpd(bestMatchOpCount);
  double estiMemSpd = memSpd * bestMatchOpCount * nThreads /
    (gate.opCount(zeroTol) * bestMatchIt->nThreads);

  std::cerr << YELLOW("Warning: ") << "No exact match to "
               "[nQubits, Precision, nThreads] = ["
            << gatenQubits << ", " << precision << ", " << nThreads
            << "] found. We estimate it by ["
            << bestMatchIt->nQubits << ", " << bestMatchIt->precision
            << ", " << bestMatchIt->nThreads << "] @ " << memSpd << " GiBps => "
               "Est. " << estiMemSpd << " GiBps.\n";

  return estiMemSpd;
}

std::ostream& StandardCostModel::display(std::ostream& os, int nLines) const {
  const int nLinesToDisplay = nLines > 0 ?
    std::min<int>(nLines, updateSpeeds.size()) :
    static_cast<int>(updateSpeeds.size());

  os << "  nQubits | Precision | nThreads | MemSpd | Norm'ed Spd \n";
  for (int i = 0; i < nLinesToDisplay; ++i) {
    int opCount = 1ULL << (2 * updateSpeeds[i].nQubits + 2);
    double memSpd = updateSpeeds[i].getMemSpd(opCount);
    double normedSpd = memSpd / opCount;
    os << "    " << std::fixed << std::setw(2) << updateSpeeds[i].nQubits
       << "    |    f" << updateSpeeds[i].precision
       << "    |    " << updateSpeeds[i].nThreads
       << "    |  " << utils::fmt_1_to_1e3(memSpd, 5)
       << " |  " << utils::fmt_1_to_1e3(normedSpd, 5)
       << "\n";
  }

  return os;
}

void PerformanceCache::saveToCSV(const std::string& fileName_) const {
  std::string fileName = fileName_;
  auto l = fileName.size();
  if (l < 4 || fileName.substr(l - 4, l) != ".csv")
    fileName += ".csv";

  std::ofstream file(fileName);
  assert(file.is_open());

  file << "nQubits,opCount,precision,irregularity,nThreads,memSpd\n";
  for (const auto&
      [nQubits, opCount, precision,
       irregularity, nThreads, memUpdateSpeed] : items) {
    file << nQubits << "," << opCount << ","
         << precision << "," << irregularity << ","
         << nThreads << ","
         << std::scientific << std::setw(6) << memUpdateSpeed << "\n";
  }
  file.close();
}

namespace {
/// @return Speed in gigabytes per second (GiBps)
double calculateMemUpdateSpeed(int nQubits, int precision, double t) {
  assert(nQubits >= 0);
  assert(precision == 32 || precision == 64);
  assert(t >= 0.0);

  return static_cast<double>(
    (precision == 32 ? 8ULL : 16ULL) << nQubits) * 1e-9 / t;
}

} // anonymous namespace

void PerformanceCache::runExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits, int nThreads, int comprehensiveness) {
  assert(nQubits >= 8 && nQubits <= 32);
  assert(comprehensiveness >= 1 && comprehensiveness <= 3);

  KernelManager kernelMgr;

  std::vector<QuantumGate> gates;
  gates.reserve(3 * nQubits);

  utils::timedExecute([&]() {
    std::random_device _rd;
    std::mt19937 gen(_rd());
    std::uniform_int_distribution<> distri(0, nQubits - 1);

    // single-qubit gates
    gates.emplace_back(QuantumGate::RandomUnitary(0));
    gates.emplace_back(QuantumGate::RandomUnitary(nQubits - 1));
    if (comprehensiveness > 2) {
      for (int q = 1; q < nQubits - 1; ++q)
        gates.emplace_back(QuantumGate::RandomUnitary(q));
    }

    // two-qubit gates
    gates.emplace_back(QuantumGate::RandomUnitary(0, 1));
    gates.emplace_back(QuantumGate::RandomUnitary(1, 2));
    gates.emplace_back(QuantumGate::RandomUnitary(nQubits - 3, nQubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(nQubits - 2, nQubits - 1));
    if (comprehensiveness > 2) {
      for (int i = 0; i < nQubits; ++i) {
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
      nQubits - 3, nQubits - 2, nQubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nQubits - 4, nQubits - 3, nQubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nQubits - 4, nQubits - 3, nQubits - 1));
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
      nQubits - 4, nQubits - 3, nQubits - 2, nQubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nQubits - 5, nQubits - 4, nQubits - 3, nQubits - 2));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nQubits - 5, nQubits - 4, nQubits - 2, nQubits - 1));
    gates.emplace_back(QuantumGate::RandomUnitary(
      nQubits - 5, nQubits - 3, nQubits - 2, nQubits - 1));
    for (int i = 0; i < 5 * comprehensiveness; ++i) {
      int a, b, c, d;
      a = distri(gen);
      do { b = distri(gen); } while (b == a);
      do { c = distri(gen); } while (c == b || c == a);
      do { d = distri(gen); } while (d == c || d == b || d == a);
      gates.emplace_back(QuantumGate::RandomUnitary(a, b, c, d));
    }
  }, "Generate gates for experiments");

  std::cerr << gates.size() << " gates generated.\n";

  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelMgr.genCPUKernel(cpuConfig, gate, "gate_" + std::to_string(i++));
  }, "Code Generation");

  utils::timedExecute([&]() {
    kernelMgr.initJIT(10, OptimizationLevel::O1, /* useLazyJIT */ false);
  }, "Initialize JIT Engine");


  timeit::Timer timer(3, 2);
  timeit::TimingResult tr;

  utils::StatevectorAlt<double> sv(nQubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (auto& kernel : kernelMgr.kernels()) {
    if (nThreads == 1)
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernel(sv.data, sv.nQubits, kernel);
      });
    else
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernelMultithread(
          sv.data, sv.nQubits, kernel, nThreads);
      });
    auto memSpd = calculateMemUpdateSpeed(nQubits, kernel.precision, tr.min);
    items.emplace_back(
      kernel.gate.nQubits(), kernel.opCount, 64,
      kernel.nLoBits, nThreads, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel.gate.qubits.begin(), kernel.gate.qubits.size()));
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

  item.nQubits = parseInt(curPtr, bufferEnd);
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
  const auto* bufferEnd = bufferBegin + bufferLength;
  file.read(bufferBegin, bufferLength);
  file.close();
  const auto* curPtr = bufferBegin;

  // parse the header
  while (*curPtr != '\n')
    ++curPtr;
  assert(std::string(bufferBegin, curPtr - bufferBegin) ==
        "nQubits,opCount,precision,irregularity,nThreads,memSpd");
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

static inline void randomRemove(QuantumGate& gate, float p) {
  auto* cMat = gate.gateMatrix.getConstantMatrix();
  assert(cMat != nullptr);

}

void PerformanceCache::runExperimentsNew(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits, int nThreads, int nRuns) {
  std::vector<std::shared_ptr<QuantumGate>> gates;
  gates.reserve(nRuns);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disFloat(0.0, 1.0);
  std::uniform_int_distribution<int> disInt(0, nQubits - 1);
  float prob = 1.0f;

  const auto randFloat = [&]() { return disFloat(gen); };
  const auto randRemove = [&](QuantumGate& gate) {
    if (prob >= 1.0)
      return;
    auto* cMat = gate.gateMatrix.getConstantMatrix();
    assert(cMat != nullptr);
    for (size_t i = 0; i < cMat->size(); ++i) {
      if (randFloat() > prob)
        cMat->data()[i].real(0.0);
      if (randFloat() > prob)
        cMat->data()[i].imag(0.0);
    }
  };

  // nQubitsWeights[q] denotes the weight for n-qubit gates
  // so length-8 array means we allow up to 7-qubit gates
  std::array<int, 8> nQubitsWeights;

  const auto addRandU1q = [&]() {
    auto a = disInt(gen);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a)));
    randRemove(*gates.back());
  };

  const auto addRandU2q = [&]() {
    int a,b;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b)));
    randRemove(*gates.back());
  };

  const auto addRandU3q = [&]() {
    int a,b,c;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c)));
    randRemove(*gates.back());
  };


  const auto addRandU4q = [&]() {
    int a,b,c,d;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d)));
    randRemove(*gates.back());
  };

  const auto addRandU5q = [&]() {
    int a,b,c,d,e;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e)));
    randRemove(*gates.back());
  };

  const auto addRandU = [&](int _nQubits) {
    switch (_nQubits) {
      case 1: addRandU1q(); break;
      case 2: addRandU2q(); break;
      case 3: addRandU3q(); break;
      case 4: addRandU4q(); break;
      case 5: addRandU5q(); break;
      default: assert(false && "Unknown nQubits");
    }
  };

  const auto randAdd = [&]() {
    int sum = 0;
    for (auto weight : nQubitsWeights)
      sum += weight;
    assert(sum > 0 && "nQubitsWeight is empty");
    std::uniform_int_distribution<int> dist(0, sum - 1);
    int r = dist(gen);
    int acc = 0;
    for (int i = 1; i < nQubitsWeights.size(); ++i) {
      acc += nQubitsWeights[i];
      if (r <= acc)
        return addRandU(i);
    }
  };

  prob = 1.0f;
  for (int n = 1; n <= 5; ++n) {
    addRandU(n);
    addRandU(n);
  }

  prob = 0.8f;
  nQubitsWeights = {0, 1, 2, 3, 5, 5, 0, 0};
  std::cerr << "nRuns = " << nRuns << std::endl;
  for (int run = gates.size(); run < nRuns; ++run)
    randAdd();

  std::cerr << "nGates = " << gates.size() << std::endl;


  KernelManager kernelMgr;
  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelMgr.genCPUKernel(cpuConfig, *gate, "gate_" + std::to_string(i++));
  }, "Code Generation");

  utils::timedExecute([&]() {
    kernelMgr.initJIT(10, OptimizationLevel::O1, /* useLazyJIT */ false);
  }, "Initialize JIT Engine");


  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  utils::StatevectorAlt<double> sv(nQubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (auto& kernel : kernelMgr.kernels()) {
    if (nThreads == 1)
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernel(sv.data, sv.nQubits, kernel);
      });
    else
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernelMultithread(
          sv.data, sv.nQubits, kernel, nThreads);
      });
    auto memSpd = calculateMemUpdateSpeed(nQubits, kernel.precision, tr.min);
    items.emplace_back(
      kernel.gate.nQubits(), kernel.opCount, 64,
      kernel.nLoBits, nThreads, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel.gate.qubits.begin(), kernel.gate.qubits.size()));
    std::cerr << ": " << memSpd << " GiBps\n";
  }
}
