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

#include <llvm/MC/TargetRegistry.h>
#include <llvm/MC/MCContext.h>


#include "llvm/Passes/PassBuilder.h"
#include "llvm/Pass.h"

#include <llvm/IR/LegacyPassManager.h>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include <cuda.h>


#define CHECK_CUDA_ERR(err) \
    if (err != CUDA_SUCCESS) {\
        std::cerr << IOColor::RED_FG << "CUDA error at line " << __LINE__ << ": " << err << "\n" << IOColor::RESET;\
        return -1; \
    }

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
    // graph.print(std::cerr) << "\n";

    auto& fusedGate = graph.getAllBlocks()[0]->quantumGate;
    auto& pMat = fusedGate->gateMatrix.getParametrizedMatrix();
    
    // printParametrizedMatrix(std::cerr, pMat);
    for (auto& p : pMat.data)
        p.removeSmallMonomials();

    // printParametrizedMatrix(std::cerr, pMat);

    // for (auto& P : fusedGate->gateMatrix.pData())
    //     P.simplify(varValues);
    // fusedGate->gateMatrix.printMatrix(std::cerr);

    IRGenerator G;
    // Function* llvmFuncPrepareParam = G.generatePrepareParameter(graph);

    // G.applyLLVMOptimization(OptimizationLevel::O3);

    auto allBlocks = graph.getAllBlocks();
    for (const auto& b : allBlocks) {
        G.generateCUDAKernel(*b->quantumGate);
    }
    // G.dumpToStderr();
    
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    errs() << "Registered platforms:\n";
    for (const auto& targ : TargetRegistry::targets())
        errs() << targ.getName() << "  " << targ.getBackendName() << "\n";
    
    auto targetTriple = Triple("nvptx64-nvidia-cuda");
    // auto targetTriple = Triple(sys::getDefaultTargetTriple());
    std::string cpu = "sm_70";
    errs() << "Target triple is: " << targetTriple.str() << "\n";

    std::string error;
    const Target *target = TargetRegistry::lookupTarget(targetTriple.str(), error);
    if (!target) {
        errs() << "Error: " << error << "\n";
        return 1;
    }
    auto* targetMachine = target->createTargetMachine(targetTriple.str(), cpu, "", {}, std::nullopt);

    G.getModule()->setTargetTriple(targetTriple.getTriple());
    G.getModule()->setDataLayout(targetMachine->createDataLayout());

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
    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

    // Optimize the IR!
    MPM.run(*G.getModule(), MAM);

    llvm::SmallString<8> data_ptx, data_ll;
    llvm::raw_svector_ostream dest_ptx(data_ptx), dest_ll(data_ll);
    dest_ptx.SetUnbuffered();
    dest_ll.SetUnbuffered();
    // print ll
    G.getModule()->print(dest_ll, nullptr);
    
    std::string ll(data_ll.begin(), data_ll.end());
    std::cerr << "===================== IR ======================\n" << ll << "\n"
              << "================== End of IR ==================\n";

    legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(passManager, dest_ptx, nullptr, CodeGenFileType::AssemblyFile)) {
        errs() << "The target machine can't emit a file of this type\n";
        return 1;
    }
    passManager.run(*G.getModule());
    std::string ptx(data_ptx.begin(), data_ptx.end());
    std::cerr << ptx << "\n";


    std::cerr << "=== Start CUDA part ===\n";

    CHECK_CUDA_ERR(cuInit(0));
    CUdevice device;
    CHECK_CUDA_ERR(cuDeviceGet(&device, 0));

    CUcontext cuCtx;
    CHECK_CUDA_ERR(cuCtxCreate(&cuCtx, 0, device));

    CUmodule cuMod;
    CHECK_CUDA_ERR(cuModuleLoadDataEx(&cuMod, ptx.c_str(), 0, nullptr, nullptr));

    CUfunction kernel;
    CHECK_CUDA_ERR(cuModuleGetFunction(&kernel, cuMod, "ptx_kernel_"));


    // wait for kernel to complete
    CHECK_CUDA_ERR(cuCtxSynchronize());

    // clean up
    CHECK_CUDA_ERR(cuModuleUnload(cuMod));
    CHECK_CUDA_ERR(cuCtxDestroy(cuCtx));

    return 0;
}