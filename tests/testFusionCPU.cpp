// #include "simulation/KernelManager.h"
// #include "simulation/JIT.h"
// #include "tests/TestKit.h"
// #include "utils/statevector.h"
//
// #include "saot/CircuitGraph.h"
// #include "saot/Parser.h"
//
// #include <filesystem>
// #include <saot/Fusion.h>
// namespace fs = std::filesystem;
//
// using namespace saot;
// using namespace utils;
//
// template<unsigned simd_s>
// static void internal() {
//   test::TestSuite suite("Fusion CPU (s = " + std::to_string(simd_s) + ")");
//
//   auto llvmContext = std::make_unique<llvm::LLVMContext>();
//   auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);
//
//   CPUKernelGenConfig kernelGenConfig;
//   kernelGenConfig.simd_s = simd_s;
//
//   CPUFusionConfig fusionConfig = CPUFusionConfig::Default;
//   NaiveCostModel costModel(3, -1, 1e-8);
//
//   std::vector<KernelInfo> kernelsBeforeFusion;
//   std::vector<QuantumGate> gatesBeforeFusion;
//   std::vector<KernelInfo> kernelsAfterFusion;
//   std::vector<QuantumGate> gatesAfterFusion;
//
//
//   std::cerr << "Test Dir " << TEST_DIR << "\n";
//   fs::path circuitDir = fs::path(TEST_DIR) / "circuits";
//   if (!fs::exists(circuitDir) || !fs::is_directory(circuitDir)) {
//     std::cerr << BOLDRED("Error: ") << "No circuit directory found\n";
//     return;
//   }
//   for (const auto& p : fs::directory_iterator(circuitDir)) {
//     if (!p.is_regular_file())
//       continue;
//
//     Parser parser(p.path().c_str());
//     auto qc = parser.parseQuantumCircuit();
//     CircuitGraph graph;
//     qc.toCircuitGraph(graph);
//     auto allBlocks = graph.getAllBlocks();
//     for (const auto& block : allBlocks) {
//       kernelsBeforeFusion.push_back(*genCPUCode(
//         *llvmModule, kernelGenConfig, *block->quantumGate,
//         "beforeFusion" + std::to_string(block->id)));
//       gatesBeforeFusion.push_back(*block->quantumGate);
//     }
//
//     applyCPUGateFusion(fusionConfig, &costModel, graph);
//     allBlocks = graph.getAllBlocks();
//     for (const auto& block : allBlocks) {
//       kernelsAfterFusion.push_back(*genCPUCode(
//         *llvmModule, kernelGenConfig, *block->quantumGate,
//         "afterFusion" + std::to_string(block->id)));
//       gatesAfterFusion.push_back(*block->quantumGate);
//     }
//
//     utils::StatevectorAlt<double> sv0(graph.nqubits, simd_s);
//     utils::StatevectorAlt<double> sv1(graph.nqubits, simd_s);
//     sv0.initialize();
//     // sv0.randomize();
//     sv1 = sv0;
//
//     auto jit = saot::createJITSession();
//     cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
//       std::move(llvmModule), std::move(llvmContext))));
//
//     for (int i = 0; i < kernelsBeforeFusion.size(); i++) {
//       auto* mPtr = gatesBeforeFusion[i].gateMatrix.getConstantMatrix()->data();
//       auto f = cantFail(
//         jit->lookup(kernelsBeforeFusion[i].llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
//       uint64_t idxEnd = 1ULL << (sv0.nqubits -
//         kernelsBeforeFusion[i].simd_s - kernelsBeforeFusion[i].qubits.size());
//       f(sv0.data, 0, idxEnd, mPtr);
//     }
//
//     for (int i = 0; i < kernelsAfterFusion.size(); i++) {
//       auto* mPtr = gatesAfterFusion[i].gateMatrix.getConstantMatrix()->data();
//       auto f = cantFail(
//         jit->lookup(kernelsAfterFusion[i].llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
//       uint64_t idxEnd = 1ULL << (sv0.nqubits -
//         kernelsAfterFusion[i].simd_s - kernelsAfterFusion[i].qubits.size());
//       f(sv1.data, 0, idxEnd, mPtr);
//     }
//
//     sv0.print();
//     std::cerr << "\n";
//     sv1.print();
//
//     gatesAfterFusion[0].gateMatrix.printCMat(std::cerr);
//
//     suite.assertAllClose(sv0.data, sv1.data, 2ULL << graph.nqubits,
//       p.path().filename(), GET_INFO());
//   }
//
//
//
//   suite.displayResult();
// }
//
// void test::test_fusionCPU() {
//   internal<1>();
//   // internal<2>();
// }