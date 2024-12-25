#include "simulation/KernelGen.h"
#include "simulation/JIT.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

using namespace saot;
using namespace utils;

template<unsigned simd_s>
static void internal() {
  test::TestSuite suite("Gate H (s = " + std::to_string(simd_s) + ")");

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;

  genCPUCode(*llvmModule, cpuConfig, QuantumGate::H(0), "gate_h_0");
  genCPUCode(*llvmModule, cpuConfig, QuantumGate::H(1), "gate_h_1");
  genCPUCode(*llvmModule, cpuConfig, QuantumGate::H(2), "gate_h_2");
  genCPUCode(*llvmModule, cpuConfig, QuantumGate::H(3), "gate_h_3");

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  auto f_h0 = cantFail(jit->lookup("gate_h_0")).toPtr<CPU_FUNC_TYPE>();
  auto f_h1 = cantFail(jit->lookup("gate_h_1")).toPtr<CPU_FUNC_TYPE>();
  auto f_h2 = cantFail(jit->lookup("gate_h_2")).toPtr<CPU_FUNC_TYPE>();
  auto f_h3 = cantFail(jit->lookup("gate_h_3")).toPtr<CPU_FUNC_TYPE>();

  StatevectorAlt<double> sv(6, simd_s);
  sv.initialize();
  suite.assertClose(sv.norm(), 1.0, "SV Initialization: Norm", GET_INFO());
  suite.assertClose(sv.prob(0), 0.0, "SV Initialization: Prob", GET_INFO());

  f_h0(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv.norm(), 1.0, "Apply H at 0: Norm", GET_INFO());
  suite.assertClose(sv.prob(0), 0.5, "Apply H at 0: Prob", GET_INFO());

  sv.initialize();
  f_h1(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv.norm(), 1.0, "Apply H at 1: Norm", GET_INFO());
  suite.assertClose(sv.prob(1), 0.5, "Apply H at 1: Prob", GET_INFO());

  sv.initialize();
  f_h2(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv.norm(), 1.0, "Apply H at 2: Norm", GET_INFO());
  suite.assertClose(sv.prob(2), 0.5, "Apply H at 2: Prob", GET_INFO());

  sv.initialize();
  f_h3(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv.norm(), 1.0, "Apply H at 3: Norm", GET_INFO());
  suite.assertClose(sv.prob(3), 0.5, "Apply H at 3: Prob", GET_INFO());

  // randomized tests
  std::vector<double> pBefore(sv.nqubits), pAfter(sv.nqubits);
  sv.randomize();
  suite.assertClose(sv.norm(), 1.0, "SV Rand Init: Norm", GET_INFO());

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h0(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[0] = pBefore[0]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 0: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 0: Prob", GET_INFO());

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h1(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[1] = pBefore[1]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 1: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 1: Prob", GET_INFO());

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h2(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[2] = pBefore[2]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 2: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 2: Prob", GET_INFO());

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h3(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simd_s), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[3] = pBefore[3]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 3: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 3: Prob", GET_INFO());

  suite.displayResult();
}

void test::test_cpuH() {
  internal<1>();
  internal<2>();
}