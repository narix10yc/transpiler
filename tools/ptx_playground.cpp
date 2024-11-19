#include "saot/Parser.h"
#include "saot/ast.h"
#include "saot/QuantumGate.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "utils/utils.h"
#include "utils/statevector.h"

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

using utils::timedExecute;
using scalar_t = double;

#define CHECK_CUDA_ERR(err) \
    if (err != CUDA_SUCCESS) {\
        std::cerr << IOColor::RED_FG << "CUDA error at line " << __LINE__ << ": " << err << "\n" << IOColor::RESET;\
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


int main(int argc, char** argv) {
    assert(argc > 1);

    openqasm::Parser parser(argv[1], 0);
    auto graph = parser.parse()->toCircuitGraph();
    // auto graph = CircuitGraph::QFT(30);

    // saot::parse::Parser parser(argv[1]);
    // auto graph = parser.parseQuantumCircuit().toCircuitGraph();

    applyCPUGateFusion(CPUFusionConfig::Default, graph);
    // graph.print(std::cerr) << "\n";

    IRGenerator G;

    CUDAGenerationConfig cudaGenConfig {
        .precision = 64,
        .useImmValues = true,
        .useConstantMemSpaceForMatPtrArg = false,
        .forceDenseKernel = false,
    };

    auto allBlocks = graph.getAllBlocks();
    for (const auto& b : allBlocks) {
        G.generateCUDAKernel(*b->quantumGate, cudaGenConfig, "kernel_block_" + std::to_string(b->id));
    }
    // G.dumpToStderr();

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
    const Target *target = TargetRegistry::lookupTarget(targetTriple.str(), error);
    if (!target) {
        errs() << "Error: " << error << "\n";
        return 1;
    }
    auto* targetMachine = target->createTargetMachine(targetTriple.str(), cpu, "", {}, std::nullopt);

    G.getModule()->setTargetTriple(targetTriple.getTriple());
    G.getModule()->setDataLayout(targetMachine->createDataLayout());

    timedExecute([&]() {
        G.applyLLVMOptimization(OptimizationLevel::O1);
    }, "Optimization Applied");

    llvm::SmallString<8> data_ptx, data_ll;
    llvm::raw_svector_ostream dest_ptx(data_ptx), dest_ll(data_ll);
    dest_ptx.SetUnbuffered();
    dest_ll.SetUnbuffered();
    // print ll
    // G.getModule()->print(dest_ll, nullptr);
    
    std::string ll(data_ll.begin(), data_ll.end());
    std::cerr << "===================== IR ======================\n" << ll << "\n"
              << "================== End of IR ==================\n";

    legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(passManager, dest_ptx, nullptr, CodeGenFileType::AssemblyFile)) {
        errs() << "The target machine can't emit a file of this type\n";
        return 1;
    }
    timedExecute([&]() {
        passManager.run(*G.getModule());
    }, "PTX Code Generated!");

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
    timedExecute([&]() {
        CHECK_CUDA_ERR(cuModuleLoadDataEx(&cuMod, ptx.c_str(), 0, nullptr, nullptr));
    }, "cuContext loaded!");

    std::vector<kernel_t> kernels(allBlocks.size());
    auto nBlocks = allBlocks.size();
    
    timedExecute([&]() {
        for (int i = 0; i < nBlocks; i++) {
            std::string kernelName = "kernel_block_" + std::to_string(allBlocks[i]->id);
            CHECK_CUDA_ERR(cuModuleGetFunction(&(kernels[i].kernel), cuMod, kernelName.c_str()));
            kernels[i].block = allBlocks[i];
            kernels[i].name = kernelName;
        }
    }, "Kernel function initialized");

    // calculate the length of matrix array needed
    unsigned lengthMatVec = 0;
    for (const auto& b : allBlocks) {
        lengthMatVec += (1 << (2 * b->nqubits() + 1));
    }

    size_t lengthSV = 2ULL * (1ULL << graph.nqubits);
    scalar_t* d_sv;
    scalar_t* d_mat;

    timedExecute([&]() {
        if (auto err = cudaMalloc((void**)(&d_sv), sizeof(scalar_t) * lengthSV))
            std::cerr << IOColor::RED_FG << "Error in cudaMalloc sv: " << err << "\n" << IOColor::RESET;
        if (auto err = cudaMalloc((void**)(&d_mat), sizeof(scalar_t) * lengthMatVec))
            std::cerr << IOColor::RED_FG << "Error in cudaMalloc mat: " << err << "\n" << IOColor::RESET;
    }, "Device memory allocated!");

    void* kernel_params[] = { &d_sv, &d_mat };

    unsigned nBlocksBits = graph.nqubits - 8;
    unsigned nThreads = 1 << 8; // 256
    
    timedExecute([&]() {
        for (int i = 0; i < kernels.size(); i++) {
            // std::cerr << "Launching kernel " << i << "\n";
            CHECK_CUDA_ERR(cuLaunchKernel(kernels[i].kernel, 
                (1 << (nBlocksBits - kernels[i].block->nqubits())), 1, 1,        // grid dim
                nThreads, 1, 1,        // block dim
                0,              // shared mem size
                0,              // stream
                kernel_params,  // kernel params
                nullptr));      // extra options
            // std::cerr << "Kernel " << i << " finished\n";

            CHECK_CUDA_ERR(cuCtxSynchronize());
        }
        cudaDeviceSynchronize();
    }, "Simulation complete!");


    timedExecute([&]() {
        for (int i = 0; i < kernels.size(); i++) {
            // std::cerr << "Launching kernel " << i << "\n";
            CHECK_CUDA_ERR(cuLaunchKernel(kernels[i].kernel, 
                (1 << (nBlocksBits - kernels[i].block->nqubits())), 1, 1,        // grid dim
                nThreads, 1, 1,        // block dim
                0,              // shared mem size
                0,              // stream
                kernel_params,  // kernel params
                nullptr));      // extra options
            // std::cerr << "Kernel " << i << " finished\n";

            CHECK_CUDA_ERR(cuCtxSynchronize());
        }
        cudaDeviceSynchronize();
    }, "Simulation complete!");

    // wait for kernel to complete
    CHECK_CUDA_ERR(cuCtxSynchronize());

    // clean up
    CHECK_CUDA_ERR(cuModuleUnload(cuMod));
    CHECK_CUDA_ERR(cuCtxDestroy(cuCtx));

    return 0;
}