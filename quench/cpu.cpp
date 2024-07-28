#include "quench/cpu.h"
#include "simulation/ir_generator.h"
#include "utils/utils.h"
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace quench::cpu;
using IRGenerator = simulation::IRGenerator;
using CircuitGraph = quench::circuit_graph::CircuitGraph;

void CodeGeneratorCPU::generate(const CircuitGraph& graph, int verbose) {
    IRGenerator irGenerator;
    const auto syncIRGeneratorConfig = [&]() {
        irGenerator.vecSizeInBits = config.simd_s;
        irGenerator.verbose = verbose;
        irGenerator.loadMatrixInEntry = config.loadMatrixInEntry;
        irGenerator.loadVectorMatrix = config.loadVectorMatrix;
        irGenerator.usePDEP = config.usePDEP;
        irGenerator.prefetchConfig.enable = config.enablePrefetch;
    };
    syncIRGeneratorConfig();
    std::string realTy;
    if (config.precision == 32) {
        irGenerator.setRealTy(IRGenerator::RealTy::Float);
        realTy = "float";
    }
    else {
        assert(config.precision == 64);
        irGenerator.setRealTy(IRGenerator::RealTy::Double);
        realTy = "double";
    }

    std::stringstream externSS;
    std::stringstream kernelSS;
    std::stringstream metaDataSS;

    externSS << "extern \"C\" {\n";

    // simulation kernel declearation
    if (config.multiThreaded)
        kernelSS << "void simulation_kernel("
                 << realTy << "* re, "
                 << realTy << "* im, "
                 << "const int nqubits, "
                 << "const int nthreads) {\n";
    else
        kernelSS << "void simulation_kernel("
                 << realTy << "* re, "
                 << realTy << "* im, "
                 << "const int nqubits) {\n";

    if (config.installTimer)
        kernelSS << "  using clock = std::chrono::high_resolution_clock;\n"
                    "  auto tic = clock::now();\n"
                    "  auto tok = clock::now();\n";

    if (config.multiThreaded)
        kernelSS << " std::vector<std::thread> threads(nthreads);\n"
                 << " uint64_t chunkSize;\n";

    kernelSS << "  uint64_t idxMax;\n";

    // apply each gate kernel
    kernelSS << "  for (const auto& data : _metaData) {\n"
             << "    idxMax = 1ULL << (nqubits - data.nqubits - S_VALUE);\n";


    if (config.multiThreaded)
        kernelSS << "    chunkSize = idxMax / nthreads;\n"
                 << "    for (unsigned i = 0; i < nthreads; i++)\n"
                 << "      threads[i] = std::thread(data.func, re, im, i*chunkSize, (i+1)*chunkSize, data.mPtr);\n"
                 << "    for (unsigned i = 0; i < nthreads; i++)\n"
                 << "      threads[i].join();\n";
    else 
        kernelSS << "    data.func(re, im, 0, idxMax, data.mPtr);\n";

    if (config.installTimer) {
        kernelSS << "    tok = clock::now();\n"
                    << "    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count() << \" ms: \" << data.info << \"\\n\";\n"
                    << "    tic = clock::now();\n";
    }

    // meta data data type
    metaDataSS << "struct _meta_data_t_ {\n"
               << "  void (*func)(" << realTy << "*, " << realTy << "*, "
               << "uint64_t, uint64_t, const void*);\n"
               << "  unsigned opCount;\n"
               << "  unsigned nqubits;\n";
    if (config.installTimer)
        metaDataSS << "  const char* info;\n";
    metaDataSS << "  const " << realTy << "* mPtr;\n"
               << "};\n"
               << "const static _meta_data_t_ _metaData[] = {\n";
    
        
    const auto allBlocks = graph.getAllBlocks();
    for (const auto& block : allBlocks) {
        const auto& gate = *(block->quantumGate);
        std::string kernelName = "kernel_block_" + std::to_string(block->id);
        irGenerator.generateKernel(gate, kernelName);
        if (config.dumpIRToMultipleFiles) {
            std::error_code ec;
            llvm::raw_fd_ostream irFile(fileName + "_ir/" + kernelName + ".ll", ec);
            irGenerator.getModule().setModuleIdentifier(kernelName + "_module");
            irGenerator.getModule().setSourceFileName(kernelName + ".ll");
            irGenerator.getModule().print(irFile, nullptr);
            irFile.close();
            irGenerator.~IRGenerator();

            new (&irGenerator) IRGenerator();
            syncIRGeneratorConfig();
        }

        externSS << " void " << kernelName << "("
                 << realTy << "*, " << realTy << "*, "
                 << "uint64_t, uint64_t, const void*);\n";
        
        // metaData
        metaDataSS << " { "
                   << "&" << kernelName << ", "
                   << block->quantumGate->opCount() << ", "
                   << block->nqubits << ", ";

        if (config.installTimer) {
            std::stringstream infoSS;
            infoSS << "block " << block->id << " ";
            utils::printVector(block->getQubits(), infoSS);
            metaDataSS << "\"" << infoSS.str() << "\", ";
        }
        
        metaDataSS << "(" << realTy << "[]){";
        for (const auto& elem : block->quantumGate->getCMatrix().data)
            metaDataSS << std::setprecision(16) << elem.real() << ","
                       << std::setprecision(16) << elem.imag() << ", ";
        metaDataSS << "} },\n";
    }
    
    externSS << "};\n";
    kernelSS << "  }\n}\n";
    metaDataSS << "};\n";

    std::ofstream hFile(fileName + ".h");
    assert(hFile.is_open());

    if (config.installTimer)
        hFile << "#include <chrono>\n"
                 "#include <iostream>\n";

    if (config.precision == 32)
        hFile << "#define USING_F32\n\n";

    if (config.multiThreaded)
        hFile << "#include <vector>\n"
                 "#include <thread>\n"
                 "#define MULTI_THREAD_SIMULATION_KERNEL\n\n";

    hFile << "#include <cstdint>\n"
          << "#define DEFAULT_NQUBITS " << graph.nqubits << "\n"
          << "#define S_VALUE " << config.simd_s << "\n"
          << externSS.str() << "\n"
          << metaDataSS.str() << "\n"
          << kernelSS.str() << "\n";

    hFile.close();

    if (!config.dumpIRToMultipleFiles) {
        std::error_code ec;
        llvm::raw_fd_ostream irFile(fileName + ".ll", ec);
        irGenerator.getModule().setModuleIdentifier(fileName + "_module");
        irGenerator.getModule().setSourceFileName(fileName + ".ll");
        irGenerator.getModule().print(irFile, nullptr);
        irFile.close();
    }


}