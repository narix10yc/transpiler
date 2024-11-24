#include "openqasm/parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "saot/Parser.h"
#include "saot/QuantumGate.h"
#include "saot/ast.h"

#include "saot/Polynomial.h"
#include "simulation/ir_generator.h"

#include <chrono>

using namespace saot;

using namespace saot::ast;
using namespace simulation;

using namespace llvm;

using utils::timedExecute;

struct kernel_t {
  Function* llvmFunc;
  void (*func)(double*, uint64_t, uint64_t, double* );
};

int main(int argc, char* *argv) {
  assert(argc > 1);

  CircuitGraph graph;
  CPUFusionConfig fusionConfig = CPUFusionConfig::Aggressive;
  IRGenerator G;
  IRGeneratorConfig irConfig;
  irConfig.simd_s = 1;
  irConfig.usePDEP = false;
  Function* llvmFuncPrepareParam;
  std::vector<kernel_t> kernels;
  void (*prepareParam)(double*, double* );

  // parse
  timedExecute(
      [&]() {
        openqasm::Parser qasmParser(argv[1], -1);
        graph = qasmParser.parse()->toCircuitGraph();
      },
      "Parsing complete!");

  // fusion
  timedExecute([&]() { applyCPUGateFusion(fusionConfig, graph); },
               "Fusion complete!");

  // ir generation
  timedExecute(
      [&]() {
        llvmFuncPrepareParam = G.generatePrepareParameter(graph);
        auto allBlocks = graph.getAllBlocks();
        std::cerr << "After fusion there are " << allBlocks.size()
                  << " blocks\n";
        kernels.reserve(allBlocks.size());
        for (const auto &b : allBlocks) {
          kernels.push_back({G.generateKernel(*b->quantumGate), nullptr});
        }
      },
      "IR generation complete!");

  // optimize
  timedExecute([&]() { G.applyLLVMOptimization(OptimizationLevel::O1); },
               "Optimization complete!");

  // jit
  timedExecute([&]() { G.createJitSession(); }, "JIT session created!");

  // function lookup
  timedExecute(
      [&]() {
        auto* jitter = G.getJitter();
        auto funcAddrOrErr = jitter->lookup(llvmFuncPrepareParam->getName());
        if (!funcAddrOrErr) {
          errs() << "Failed to look up function\n"
                 << funcAddrOrErr.takeError() << "\n";
        }

        prepareParam = funcAddrOrErr->toPtr<void(double*, double* )>();
        for (auto &kernel : kernels) {
          kernel.func =
              cantFail(jitter->lookup(llvmFuncPrepareParam->getName()))
                  .toPtr<void(double*, uint64_t, uint64_t, double* )>();
        }
      },
      "Function found");

  return 0;
}