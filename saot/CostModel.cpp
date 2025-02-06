#include "saot/CostModel.h"
#include "saot/QuantumGate.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>
#include "utils/statevector.h"

using namespace saot;
using namespace llvm;

double NaiveCostModel::computeSpeed(
    const QuantumGate& gate, int precision, int nThreads) const {
  if (gate.nQubits() > maxnQubits)
    return 1e-8;
  if (gate.opCount(zeroTol) > maxOp)
    return 1e-8;

  return 1.0;
}

CostResult NaiveCostModel::computeBenefit(
    const QuantumGate& lhsGate, const QuantumGate& rhsGate,
    CircuitGraphContext& context) const {
  assert(false && "Not Implemented");
  return { 0.0, nullptr };

  // auto cQubits = lhsGate.qubits;
  // for (const auto q : rhsGate.qubits) {
  //   if (std::ranges::find(cQubits, q) == cQubits.end())
  //     cQubits.push_back(q);
  // }
  //
  // // check fusion eligibility: nQubits
  // if (cQubits.size() > this->maxnQubits) {
  //   // std::cerr << CYAN("Rejected due to maxnQubits\n");
  //   return { 0.0, nullptr };
  // }
  //
  // // check fusion eligibility: opCount
  // auto cGate = lhsGate.lmatmul(rhsGate);
  // if (maxOp > 0 && cGate.opCount(zeroTol) > maxOp) {
  //   // std::cerr << CYAN("Rejected due to OpCount\n");
  //   return { 0.0, nullptr };
  // }
  //
  // // accept candidate
  // // std::cerr << GREEN("Fusion accepted!\n");
  // return { 1.0, cGate };
}

CostResult AdaptiveCostModel::computeBenefit(
    const QuantumGate &lhsGate, const QuantumGate &rhsGate,
    CircuitGraphContext &context) const {
  assert(0 && "Not Implemented");
  return {0.0, nullptr};
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

CostResult StandardCostModel::computeBenefit(
    const QuantumGate& lhsGate, const QuantumGate& rhsGate,
    CircuitGraphContext& context) const {
  assert(0 && "Not Implemented");
  return {0.0, nullptr};
  // auto cQubits = lhsGate.qubits;
  // for (const auto q : rhsGate.qubits) {
  //   if (std::ranges::find(cQubits, q) == cQubits.end())
  //     cQubits.push_back(q);
  // }
  //
  // auto* cGate = context.quantumGatePool.acquire(rhsGate.lmatmul(lhsGate));
  // const auto lSpd = computeSpeed(lhsGate, TODO, TODO);
  // const auto rSpd = computeSpeed(rhsGate, TODO, TODO);
  // const auto cSpd = computeSpeed(*cGate, TODO, TODO);
  // double benefit = 2 * cSpd / (lSpd + rSpd) - 1;
  //
  // std::cerr << "lSpd = " << lSpd << ", rSpd = " << rSpd
  //           << ", cSpd = " << cSpd << "; Benefit = " << benefit << "\n";
  //
  // return { benefit, cGate };
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

  return static_cast<double>((precision == 32 ? 8ULL : 16ULL) << nQubits) *
    1e-9 / t;
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
  std::vector<std::unique_ptr<KernelInfo>> kernelInfos;
  kernelInfos.reserve(3 * nQubits);

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

  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelMgr.genCPUKernel(cpuConfig, gate, "gate_" + std::to_string(i++));
  }, "Code Generation");


  timeit::Timer timer;
  timeit::TimingResult tr;

  utils::StatevectorAlt<double> sv(nQubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (unsigned i = 0, s = kernelInfos.size(); i < s; ++i) {
    const auto& kernel = kernelInfos[i];
    if (nThreads == 1)
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernel(sv.data, sv.nQubits, kernel->llvmFuncName);
      });
    else
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernelMultithread(
          sv.data, sv.nQubits, kernel->llvmFuncName, nThreads);
      });
    auto memSpd = calculateMemUpdateSpeed(nQubits, kernel->precision, tr.min);
    items.emplace_back(
      kernel->gate.nQubits(), kernel->opCount, 64,
      kernel->nLoBits, nThreads, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel->gate.qubits.begin(), kernel->gate.qubits.size()));
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
  auto* bufferEnd = bufferBegin + bufferLength;
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