#include "saot/Parser.h"
#include "saot/ast.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"

#include "openqasm/parser.h"

#include "simulation/ir_generator.h"


#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/TargetParser/Host.h>

#include <llvm/MC/TargetRegistry.h>
#include <llvm/MC/MCContext.h>

#include <llvm/IR/LegacyPassManager.h>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include <cuda.h>
#include <cuda_runtime.h>


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
    assert(argc > 1);

    openqasm::Parser parser(argv[1], 0);
    auto graph = parser.parse()->toCircuitGraph();

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


    CUDAGenerationConfig cudaGenConfig {
        .useImmValues = true,
        .useConstantMemSpaceForMatPtrArg = false
    };

    auto allBlocks = graph.getAllBlocks();
    for (const auto& b : allBlocks) {
        G.generateCUDAKernel(*b->quantumGate, cudaGenConfig);
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

    G.applyLLVMOptimization(OptimizationLevel::O2);


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

    std::vector<double> svHost { 1.0, 0.0, 0.0, 0.0 };
    double* svDevice;
    std::vector<double> matHost { M_SQRT1_2, 0.0, M_SQRT1_2, 0.0, M_SQRT1_2, 0.0, -M_SQRT1_2, 0.0 };
    double* matDevice;

    cudaError_t err;

    err = cudaMalloc((void**)(&svDevice), sizeof(double) * svHost.size());
    if (err) {
        std::cerr << IOColor::RED_FG << "Error in cudaMalloc sv: " << err << "\n" << IOColor::RESET;
        return 1;
    }
    err = cudaMalloc((void**)(&matDevice), sizeof(double) * matHost.size());
    if (err) {
        std::cerr << IOColor::RED_FG << "Error in cudaMalloc mat: " << err << "\n" << IOColor::RESET;
        return 1;
    }

    err = cudaMemcpy((void*)svDevice, static_cast<const void*>(svHost.data()), sizeof(double) * svHost.size(), cudaMemcpyHostToDevice);
    if (err) {
        std::cerr << IOColor::RED_FG << "Error in host => device memCpy: " << err << "\n" << IOColor::RESET;
        return 1;
    }
    err = cudaMemcpy((void*)matDevice, static_cast<const void*>(matHost.data()), sizeof(double) * matHost.size(), cudaMemcpyHostToDevice);
    if (err) {
    
        std::cerr << IOColor::RED_FG << "Error in host => device memCpy: " << err << "\n" << IOColor::RESET;
        return 1;
    }

    void* kernel_params[] = { &svDevice, &matDevice };
    CHECK_CUDA_ERR(cuLaunchKernel(kernel, 
        1, 1, 1,        // grid dim
        1, 1, 1,        // block dim
        0,              // shared mem size
        0,              // stream
        kernel_params,  // kernel params
        nullptr));      // extra options

    err = cudaMemcpy(svHost.data(), svDevice, sizeof(double) * svHost.size(), cudaMemcpyDeviceToHost);
    if (err) {
        std::cerr << IOColor::RED_FG << "Error in device => host memCpy: " << err << "\n" << IOColor::RESET;
        return 1;
    }
    err = cudaMemcpy(matHost.data(), matDevice, sizeof(double) * matHost.size(), cudaMemcpyDeviceToHost);
    if (err) {
        std::cerr << IOColor::RED_FG << "Error in device => host memCpy: " << err << "\n" << IOColor::RESET;
        return 1;
    }

    utils::printVector(svHost) << "\n";

    // wait for kernel to complete
    CHECK_CUDA_ERR(cuCtxSynchronize());

    // clean up
    CHECK_CUDA_ERR(cuModuleUnload(cuMod));
    CHECK_CUDA_ERR(cuCtxDestroy(cuCtx));

    return 0;
}