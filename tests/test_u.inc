/// @brief Test general single-qubit unitary gates
template<unsigned simd_s>
void test_U() {
  TestSuite suite("Gate U with simd_s = " + std::to_string(simd_s));
  constexpr int nqubits = 5;
  StatevectorAlt<double, simd_s> sv0(nqubits), sv1(nqubits);
  sv0.randomize();
  suite.assertClose(sv0.norm(), 1.0, GET_INFO("Rand SV: Norm"));
  sv1 = sv0;

  auto randMatrix = utils::randomUnitaryMatrix(2);

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;

  QuantumGate gate0(GateMatrix(utils::randomUnitaryMatrix(2)), 0);
  QuantumGate gate1(GateMatrix(utils::randomUnitaryMatrix(2)), 1);
  QuantumGate gate2(GateMatrix(utils::randomUnitaryMatrix(2)), 2);
  QuantumGate gate3(GateMatrix(utils::randomUnitaryMatrix(2)), 3);

  genCPUCode(*llvmModule, cpuConfig, gate0, "gate_u_0");
  genCPUCode(*llvmModule, cpuConfig, gate1, "gate_u_1");
  genCPUCode(*llvmModule, cpuConfig, gate2, "gate_u_2");
  genCPUCode(*llvmModule, cpuConfig, gate3, "gate_u_3");

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  auto f_h0 = jit->lookup("gate_u_0")->toPtr<FUNC_TYPE>();
  auto f_h1 = jit->lookup("gate_u_1")->toPtr<FUNC_TYPE>();
  auto f_h2 = jit->lookup("gate_u_2")->toPtr<FUNC_TYPE>();
  auto f_h3 = jit->lookup("gate_u_3")->toPtr<FUNC_TYPE>();

  f_h0(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv0.norm(), 1.0, GET_INFO("Apply U at 0: Norm"));
  sv1.applyGate(gate0);
  suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, GET_INFO("Apply U at 0: Amplitudes"));

  f_h1(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv0.norm(), 1.0, GET_INFO("Apply U at 1: Norm"));
  sv1.applyGate(gate1);
  suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, GET_INFO("Apply U at 1: Amplitudes"));

  f_h2(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv0.norm(), 1.0, GET_INFO("Apply U at 2: Norm"));
  sv1.applyGate(gate2);
  suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, GET_INFO("Apply U at 2: Amplitudes"));

  f_h3(sv0.data, 0ULL, 1ULL << (nqubits - 1 - cpuConfig.simd_s), nullptr);
  suite.assertClose(sv0.norm(), 1.0, GET_INFO("Apply U at 3: Norm"));
  sv1.applyGate(gate3);
  suite.assertAllClose(sv0.data, sv1.data, 2ULL << nqubits, GET_INFO("Apply U at 3: Amplitudes"));

  suite.displayResult();
}