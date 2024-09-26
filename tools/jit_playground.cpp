#include "saot/parser.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"

#include "saot/Polynomial.h"
#include "simulation/ir_generator.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Pass.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"



using namespace saot;

using namespace saot::ast;
using namespace saot::quantum_gate;
using namespace saot::circuit_graph;
using namespace simulation;

using namespace llvm;

int main(int argc, char** argv) {
    std::vector<std::pair<int, double>> varValues {
        {0, 1.1}, {1, 0.4}, {2, 0.1}, {3, -0.3}, {4, -0.9}, {5, 1.9}};


    assert(argc > 1);

    Parser parser(argv[1]);
    auto qc = parser.parse();
    std::cerr << "Recovered:\n";

    std::ofstream file(std::string(argv[1]) + ".rec");
    qc.print(file);

    auto graph = qc.toCircuitGraph();
    graph.updateFusionConfig(FusionConfig::Default());
    graph.greedyGateFusion();

    auto* fusedGate = graph.getAllBlocks()[0]->quantumGate.get();

    fusedGate->gateMatrix.printMatrix(std::cerr) << "\n";

    for (auto& P : fusedGate->gateMatrix.pData()) {
        P.removeSmallMonomials();
    }
    fusedGate->gateMatrix.printMatrix(std::cerr) << "\n";

    // for (auto& P : fusedGate->gateMatrix.pData())
        // P.simplify(varValues);
    fusedGate->gateMatrix.printMatrix(std::cerr);

    IRGenerator G;
    Function* func = G.generatePrepareParameter(graph);
    G.dumpToStderr();

    // optimize
    // Create the analysis managers.
    // These must be declared in this order so that they are destroyed in the
    // correct order due to inter-analysis-manager references.
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    PassBuilder PB;

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    // This one corresponds to a typical -O2 optimization pipeline.
    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);

    // Optimize the IR!
    MPM.run(*G.getModule(), MAM);

    G.dumpToStderr();

    // JIT
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    auto jit = cantFail(orc::LLJITBuilder().create());
    auto TSModule = orc::ThreadSafeModule(std::move(G._module), std::move(G._context));

    if (auto err = jit->addIRModule(std::move(TSModule))) {
        errs() << "Error adding module to JIT: " << err;
        return 1;
    }

    auto jitFunc = jit->lookup(func->getName());

    if (!jitFunc) {
        std::cerr << Color::RED_FG << "Error: " << Color::RESET << "JIT Error cannot be found!\n";
        return 1;
    }
    

    return 0;
}