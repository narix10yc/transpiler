#include "saot/Parser.h"
#include "saot/ast.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "saot/Polynomial.h"
#include "simulation/jit.h"

using namespace saot;

using namespace saot::ast;
using namespace simulation;

using namespace llvm;

int main(int argc, char** argv) {
    std::vector<std::pair<int, double>> varValues {
        {0, 1.1}, {1, 0.4}, {2, 0.1}, {3, -0.3}, {4, -0.9}, {5, 1.9}};


    assert(argc > 1);

    parse::Parser parser(argv[1]);
    auto qc = parser.parseQuantumCircuit();
    std::cerr << "Recovered:\n";

    std::ofstream file(std::string(argv[1]) + ".rec");
    qc.print(file);

    auto graph = qc.toCircuitGraph();
    graph.print(std::cerr) << "\n";

    applyCPUGateFusion(CPUFusionConfig::Default, graph);
    graph.print(std::cerr) << "\n";

    auto& fusedGate = graph.getAllBlocks()[0]->quantumGate;

    auto& pMat = fusedGate->gateMatrix.getParametrizedMatrix();
    
    // printParametrizedMatrix(std::cerr, pMat);
    for (auto& p : pMat.data)
        p.removeSmallMonomials();

    printParametrizedMatrix(std::cerr, pMat);

    // for (auto& P : fusedGate->gateMatrix.pData())
    //     P.simplify(varValues);
    // fusedGate->gateMatrix.printMatrix(std::cerr);

    IRGenerator G;
    Function* llvmFuncPrepareParam = G.generatePrepareParameter(graph);
    auto allBlocks = graph.getAllBlocks();
    for (const auto& b : allBlocks) {
        G.generateKernel(*b->quantumGate);
    }
    G.dumpToStderr();

    // G.applyLLVMOptimization(OptimizationLevel::O2);
    // G.dumpToStderr();

    // JIT
    jit::JitEngine jitter(G);

    auto funcAddrOrErr = jitter.JIT->lookup(llvmFuncPrepareParam->getName());
    if (!funcAddrOrErr) {
        errs() << "Failed to look up function\n" << funcAddrOrErr.takeError() << "\n";
        return 1;
    }

    auto prepareParameter = funcAddrOrErr->toPtr<void(double*, double*)>();

    std::vector<double> circuitParameters { 1.344, 3.109, 0.12 };
    std::vector<double> circuitMatrices(8);

    prepareParameter(circuitParameters.data(), circuitMatrices.data());

    utils::printVector(circuitMatrices);



    
    


    return 0;
}