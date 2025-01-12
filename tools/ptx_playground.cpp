#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "saot/Parser.h"
#include "saot/QuantumGate.h"
#include "saot/ast.h"
#include "utils/statevector.h"
#include "utils/utils.h"

#include "openqasm/parser.h"

#include "simulation/ir_generator.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

#include <llvm/MC/MCContext.h>
#include <llvm/MC/TargetRegistry.h>

#include <llvm/IR/LegacyPassManager.h>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include <cuda.h>
#include <cuda_runtime.h>

using utils::timedExecute;
using scalar_t = float;

#define CHECK_CUDA_ERR(err)                                                    \
  if (err != CUDA_SUCCESS) {                                                   \
    std::cerr << IOColor::RED_FG << "CUDA error at line " << __LINE__ << ": "  \
              << err << "\n"                                                   \
              << IOColor::RESET;                                               \
  }

using namespace saot;

using namespace saot::ast;
using namespace simulation;

using namespace llvm;

struct kernel_t {
  GateBlock* block;
  std::string name;
  CUfunction kernel;
};

static CircuitGraph& getCircuitH1(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("h");
  for (int q = 0; q < nqubits; q++)
    graph.addGate(mat, {q});

  return graph;
}

static CircuitGraph& getCircuitU1(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
  for (int q = 0; q < nqubits; q++)
    graph.addGate(mat, {q});

  return graph;
}

static CircuitGraph& getCircuitH2(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::MatrixH_c;
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitU2(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitH3(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("h");
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 2) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitZ3(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("z");
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 2) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitU3(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 2) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitH4(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("h");
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 2) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 3) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

static CircuitGraph& getCircuitZ4(CircuitGraph& graph, int nqubits) {
  auto mat = GateMatrix::FromName("z");
  for (int q = 0; q < nqubits; q++) {
    QuantumGate gate(mat, {q});
    gate = gate.lmatmul(QuantumGate(mat, {(q + 1) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 2) % nqubits}));
    gate = gate.lmatmul(QuantumGate(mat, {(q + 3) % nqubits}));
    graph.addGate(gate);
  }

  return graph;
}

int main(int argc, const char** argv) {
  assert(argc > 1);

  // openqasm::Parser parser(argv[1], 0);
  // auto graph = parser.parse()->toCircuitGraph();
  auto graph = CircuitGraph::QFTCircuit(30);

  // CircuitGraph graph;
  // getCircuitH1(graph, 1);
  // for (int i = 0; i < 20; i++)
  // getCircuitH2(graph, 28);

  // saot::parse::Parser parser(argv[1]);
  // auto graph = parser.parseQuantumCircuit().toCircuitGraph();

  auto fusionConfig = CPUFusionConfig::Default;
  // CPUFusionConfig fusionConfig = CPUFusionConfig {
  //     .maxNQubits = 3,
  //     .maxOpCount = 9999,
  //     .zeroSkippingThreshold = 1e-8,
  //     .allowMultipleTraverse = true,
  //     .incrementScheme = true,
  // };
  applyCPUGateFusion(fusionConfig, graph);

  // graph.print(std::cerr) << "\n";

  IRGenerator G;

  CUDAGenerationConfig cudaGenConfig{
      .precision = 32,
      .useImmValues = true,
      .useConstantMemSpaceForMatPtrArg = false,
      .forceDenseKernel = false,
  };

  auto allBlocks = graph.getAllBlocks();
  for (const auto& b : allBlocks)
    G.generateCUDAKernel(*b->quantumGate, cudaGenConfig,
                         "kernel_block_" + std::to_string(b->id));

  // auto mat = GateMatrix::MatrixI1_c;
  // GateMatrix::c_matrix_t mat {
  //     {0.5, 0.0}, {0.0, 0.0},
  //     {0.0, 0.0}, {0.5, 0.0}
  // };
  // QuantumGate gate(mat, {0});
  // gate = gate.lmatmul(QuantumGate(mat, {3}));
  // gate = gate.lmatmul(QuantumGate(mat, {4}));

  // G.generateCUDAKernel(gate, cudaGenConfig, "test_kernel");

  std::cerr << "There are " << graph.countBlocks() << " blocks after fusion\n";

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // errs() << "Registered platforms:\n";
  // for (const auto& targ : TargetRegistry::targets())
  //     errs() << targ.getName() << "  " << targ.getBackendName() << "\n";

  auto targetTriple = Triple("nvptx64-nvidia-cuda");
  // auto targetTriple = Triple(sys::getDefaultTargetTriple());
  std::string cpu = "sm_70";
  errs() << "Target triple is: " << targetTriple.str() << "\n";

  std::string error;
  const Target* target =
      TargetRegistry::lookupTarget(targetTriple.str(), error);
  if (!target) {
    errs() << "Error: " << error << "\n";
    return 1;
  }
  auto* targetMachine = target->createTargetMachine(targetTriple.str(), cpu, "",
                                                    {}, std::nullopt);

  G.getModule()->setTargetTriple(targetTriple.getTriple());
  G.getModule()->setDataLayout(targetMachine->createDataLayout());

  timedExecute([&]() { G.applyLLVMOptimization(OptimizationLevel::O1); },
               "Optimization Applied");
  // G.dumpToStderr();

  llvm::SmallString<8> data_ptx;
  llvm::raw_svector_ostream dest_ptx(data_ptx);
  dest_ptx.SetUnbuffered();

  legacy::PassManager passManager;
  if (targetMachine->addPassesToEmitFile(passManager, dest_ptx, nullptr,
                                         CodeGenFileType::AssemblyFile)) {
    errs() << "The target machine can't emit a file of this type\n";
    return 1;
  }
  timedExecute([&]() { passManager.run(*G.getModule()); },
               "PTX Code Generated!");

  std::string ptx(data_ptx.begin(), data_ptx.end());
  // std::cerr << ptx << "\n";

  std::cerr << "=== Start CUDA part ===\n";

  CHECK_CUDA_ERR(cuInit(0));
  CUdevice device;
  CHECK_CUDA_ERR(cuDeviceGet(&device, 0));

  CUcontext cuCtx;
  CHECK_CUDA_ERR(cuCtxCreate(&cuCtx, 0, device));

  CUmodule cuMod;

  std::cerr << "Loading PTX into cuda context\n";
  timedExecute(
      [&]() {
        CHECK_CUDA_ERR(
            cuModuleLoadDataEx(&cuMod, ptx.c_str(), 0, nullptr, nullptr));
      },
      "cuContext loaded!");

  std::vector<kernel_t> kernels(allBlocks.size());
  auto nBlocks = allBlocks.size();

  // CHECK_CUDA_ERR(cuModuleGetFunction(&(kernels[0].kernel), cuMod,
  // "test_kernel"));

  timedExecute(
      [&]() {
        for (int i = 0; i < nBlocks; i++) {
          std::string kernelName =
              "kernel_block_" + std::to_string(allBlocks[i]->id);
          CHECK_CUDA_ERR(cuModuleGetFunction(&(kernels[i].kernel), cuMod,
                                             kernelName.c_str()));
          kernels[i].block = allBlocks[i];
          kernels[i].name = kernelName;
        }
      },
      "Kernel function initialized");

  // calculate the length of matrix array needed
  unsigned lengthMatVec = 0;
  for (const auto& b : allBlocks) {
    lengthMatVec += (1 << (2 * b->nqubits() + 1));
  }

  size_t lengthSV = 2ULL * (1ULL << graph.nqubits);
  scalar_t* d_sv;
  scalar_t* d_mat;

  timedExecute(
      [&]() {
        if (auto err =
                cudaMalloc((void* *)(&d_sv), sizeof(scalar_t) * lengthSV))
          std::cerr << IOColor::RED_FG << "Error in cudaMalloc sv: " << err
                    << "\n"
                    << IOColor::RESET;
        if (auto err =
                cudaMalloc((void* *)(&d_mat), sizeof(scalar_t) * lengthMatVec))
          std::cerr << IOColor::RED_FG << "Error in cudaMalloc mat: " << err
                    << "\n"
                    << IOColor::RESET;
      },
      "Device memory allocated!");

  void* kernel_params[] = {&d_sv, &d_mat};
  // void* kernel_params[] = { &d_sv };

  unsigned nBlocksBits = graph.nqubits - 8;
  unsigned nThreads = 1 << 8; // 256

  // timedExecute([&]() {
  //     for (int i = 0; i < 100; i++) {
  //         CHECK_CUDA_ERR(cuLaunchKernel(kernels[0].kernel,
  //             (1 << (30 - 8)), 1, 1,        // grid dim
  //             nThreads, 1, 1,        // block dim
  //             0,              // shared mem size
  //             0,              // stream
  //             kernel_params,  // kernel params
  //             nullptr));      // extra options
  //         CHECK_CUDA_ERR(cuCtxSynchronize());
  //     }
  // }, "Incremented");

  timedExecute(
      [&]() {
        // for (int r = 0; r < 5; r++) {
        for (int i = 0; i < kernels.size(); i++) {
          // std::cerr << "Launching kernel " << i << "\n";
          CHECK_CUDA_ERR(
              cuLaunchKernel(kernels[i].kernel,
                             (1 << (nBlocksBits - kernels[i].block->nqubits())),
                             1, 1,           // grid dim
                             nThreads, 1, 1, // block dim
                             0,              // shared mem sizecd
                             0,              // stream
                             kernel_params,  // kernel params
                             nullptr));      // extra options
          CHECK_CUDA_ERR(cuCtxSynchronize());
        }
        // }
      },
      "Simulation complete!");

  timedExecute(
      [&]() {
        // for (int r = 0; r < 5; r++) {
        for (int i = 0; i < kernels.size(); i++) {
          // std::cerr << "Launching kernel " << i << "\n";
          CHECK_CUDA_ERR(
              cuLaunchKernel(kernels[i].kernel,
                             (1 << (nBlocksBits - kernels[i].block->nqubits())),
                             1, 1,           // grid dim
                             nThreads, 1, 1, // block dim
                             0,              // shared mem sizecd
                             0,              // stream
                             kernel_params,  // kernel params
                             nullptr));      // extra options
          CHECK_CUDA_ERR(cuCtxSynchronize());
        }
        // }
      },
      "Simulation complete!");

  // wait for kernel to complete
  CHECK_CUDA_ERR(cuCtxSynchronize());

  // clean up
  CHECK_CUDA_ERR(cuModuleUnload(cuMod));
  CHECK_CUDA_ERR(cuCtxDestroy(cuCtx));

  return 0;
}