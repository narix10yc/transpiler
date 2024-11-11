#include "saot/Parser.h"
#include "saot/ast.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "saot/Polynomial.h"
#include "simulation/jit.h"


#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/TargetParser/Host.h>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"


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
    // Function* llvmFuncPrepareParam = G.generatePrepareParameter(graph);

    G.applyLLVMOptimization(OptimizationLevel::O3);

    auto allBlocks = graph.getAllBlocks();
    for (const auto& b : allBlocks) {
        G.generateCUDAKernel(*b->quantumGate);
    }
    G.dumpToStderr();
    


    return 0;
}