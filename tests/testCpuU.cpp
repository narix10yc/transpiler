#include "simulation/KernelGen.h"
#include "simulation/JIT.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"
#include <random>

using namespace saot;

template<unsigned simd_s, unsigned nqubits>
static void internal_U1q() {
  test::TestSuite suite(
    "Gate U1q (s=" + std::to_string(simd_s) +
    ", n=" + std::to_string(nqubits) + ")");
  utils::StatevectorAlt<double> sv0(nqubits, simd_s), sv1(nqubits, simd_s);

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  std::vector<QuantumGate> gateImm(nqubits), gateLoad(nqubits);
  std::vector<KernelMetadata> kernelImm(nqubits), kernelLoad(nqubits);

  // kernels with imm value matrix
  for (unsigned q = 0; q < nqubits; q++) {
    gateImm[q] = QuantumGate(GateMatrix(utils::randomUnitaryMatrix(2)), q);
    kernelImm[q].quantumGate = &gateImm[q];
    kernelImm[q].llvmFuncName = "gateImm_" + std::to_string(q);

    gateLoad[q] = QuantumGate(GateMatrix(utils::randomUnitaryMatrix(2)), q);
    kernelLoad[q].quantumGate = &gateLoad[q];
    kernelLoad[q].llvmFuncName = "gateLoad_" + std::to_string(q);
  }

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::UseMatImmValues;
  for (const auto& item : kernelImm)
    genCPUCode(*llvmModule, cpuConfig, *item.quantumGate, item.llvmFuncName);
  cpuConfig.forceDenseKernel = true;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;
  for (const auto& item : kernelLoad)
    genCPUCode(*llvmModule, cpuConfig, *item.quantumGate, item.llvmFuncName);

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  for (unsigned q = 0; q < nqubits; q++) {
    sv0.randomize();
    sv1 = sv0;
    auto f = cantFail(jit->lookup(kernelImm[q].llvmFuncName)).toPtr<CPU_FUNC_TYPE>();
    f(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s), nullptr);
    suite.assertClose(sv0.norm(), 1.0,
      "Apply U1q (imm matrix) at " + std::to_string(q) + ": Norm", GET_INFO());
    sv1.applyGate(*kernelImm[q].quantumGate);
    suite.assertClose(utils::fidelity(sv0, sv1), 1.0,
      "Apply U1q (imm matrix) at " + std::to_string(q) + ": Fidelity",
      GET_INFO());
  }

  for (unsigned q = 0; q < nqubits; q++) {
    sv0.randomize();
    sv1 = sv0;
    auto f = cantFail(jit->lookup(kernelLoad[q].llvmFuncName)).toPtr<CPU_FUNC_TYPE>();
    f(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s),
      kernelLoad[q].quantumGate->gateMatrix.getConstantMatrix()->data());
    suite.assertClose(sv0.norm(), 1.0,
      "Apply U1q (loaded matrix) at " + std::to_string(q) + ": Norm", GET_INFO());
    sv1.applyGate(*kernelLoad[q].quantumGate);
    suite.assertClose(utils::fidelity(sv0, sv1), 1.0,
      "Apply U1q (loaded matrix) at " + std::to_string(q) + ": Fidelity",
      GET_INFO());
  }

  suite.displayResult();
}

template<unsigned simd_s, unsigned nqubits>
static void internal_U2q() {
  test::TestSuite suite(
    "Gate U2q (s=" + std::to_string(simd_s) +
    ", n=" + std::to_string(nqubits) + ")");
  utils::StatevectorAlt<double>
  sv0(nqubits, simd_s), sv1(nqubits, simd_s), sv2(nqubits, simd_s);

  const auto randomizeSV = [&sv0, &sv1, &sv2]() {
    sv0.randomize();
    sv1 = sv0;
    sv2 = sv0;
  };

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  std::vector<QuantumGate> gates(nqubits);
  std::vector<KernelMetadata> kernelImm(nqubits), kernelLoad(nqubits);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nqubits - 1);

  // generate random gates, set up kernel names
  for (unsigned i = 0; i < nqubits; i++) {
    int a, b;
    a = d(gen);
    do { b = d(gen); } while (b == a);
    gates[i] = QuantumGate(GateMatrix(utils::randomUnitaryMatrix(4)), {a, b});
    kernelImm[i].quantumGate = &gates[i];
    kernelImm[i].llvmFuncName = "gateImm_" + std::to_string(i);
    kernelLoad[i].quantumGate = &gates[i];
    kernelLoad[i].llvmFuncName = "gateLoad_" + std::to_string(i);
  }

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::UseMatImmValues;
  for (const auto& item : kernelImm)
    genCPUCode(*llvmModule, cpuConfig, *item.quantumGate, item.llvmFuncName);
  cpuConfig.forceDenseKernel = true;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;
  for (const auto& item : kernelLoad)
    genCPUCode(*llvmModule, cpuConfig, *item.quantumGate, item.llvmFuncName);

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  for (unsigned i = 0; i < nqubits; i++) {
    randomizeSV();
    int a = gates[i].qubits[0];
    int b = gates[i].qubits[1];
    auto* mPtr = gates[i].gateMatrix.getConstantMatrix()->data();
    auto fImm =
      cantFail(jit->lookup(kernelImm[i].llvmFuncName)).toPtr<CPU_FUNC_TYPE>();
    auto fLoad =
      cantFail(jit->lookup(kernelLoad[i].llvmFuncName)).toPtr<CPU_FUNC_TYPE>();
    std::stringstream ss;
    ss << "Apply U2q at " << a << " and " << b;
    fImm(sv0.data, 0ULL, 1ULL << (nqubits - 2 - cpuConfig.simd_s), nullptr);
    fLoad(sv1.data, 0ULL, 1ULL << (nqubits - 2 - cpuConfig.simd_s), mPtr);
    sv2.applyGate(gates[i]);
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    suite.assertClose(utils::fidelity(sv0, sv2), 1.0,
      ss.str() + ": Imm Fidelity", GET_INFO());
    suite.assertClose(utils::fidelity(sv1, sv2), 1.0,
      ss.str() + ": Load Fidelity", GET_INFO());
  }

  suite.displayResult();
}

void test::test_cpuU() {
  internal_U1q<1, 8>();
  internal_U1q<2, 12>();
  internal_U2q<1, 8>();
  internal_U2q<2, 8>();
}