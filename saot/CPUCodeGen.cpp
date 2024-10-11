#include "saot/cpu.h"

#include "utils/utils.h"
#include "utils/iocolor.h"

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace IOColor;
using namespace saot;
using namespace simulation;

std::ostream&
CodeGeneratorCPUConfig::display(int verbose, std::ostream& os) const {
    os << CYAN_FG << BOLD << "=== CodeGen Configuration ===\n" << RESET;
    
    os << "Multi-threading "
        << ((multiThreaded) ? "enabled" : "disabled")
        << ".\n";
    
    if (installTimer)
        os << "Timer installed\n";
    
    os << CYAN_FG << "Detailed IR settings:\n" << RESET;
    irConfig.display(verbose, false, os);

    os << CYAN_FG << BOLD << "=============================\n" << RESET;
    return os;
}

void CodeGeneratorCPU::generate(
        const CircuitGraph& graph, int debugLevel, bool forceInOrder) {
    IRGenerator irGenerator(config.irConfig);

    bool isSepKernel = (config.irConfig.ampFormat == IRGeneratorConfig::SepFormat);

    const std::string realTy = (config.irConfig.precision == 32) ? "float" : "double";

    std::stringstream externSS;
    std::stringstream kernelSS;
    std::stringstream metaDataSS;

    externSS << "extern \"C\" {\n";

    // simulation kernel declearation
    kernelSS << "void simulation_kernel(";
    if (isSepKernel)
        kernelSS << realTy << "* re, " << realTy << "* im, ";
    else
        kernelSS << realTy << "* sv, ";
        
    kernelSS << "const int nqubits";
    if (config.multiThreaded)
        kernelSS << ", const int nthreads";
    kernelSS << ") {\n";

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
             << "    idxMax = 1ULL << (nqubits - data.nqubits - SIMD_S);\n";

    std::string sv_arg = (isSepKernel) ? "re, im" : "sv";
    if (config.multiThreaded)
        kernelSS << "    chunkSize = idxMax / nthreads;\n"
                 << "    for (unsigned i = 0; i < nthreads; i++)\n"
                 << "      threads[i] = std::thread(data.func, " << sv_arg << ", i*chunkSize, (i+1)*chunkSize, data.mPtr);\n"
                 << "    for (unsigned i = 0; i < nthreads; i++)\n"
                 << "      threads[i].join();\n";
    else 
        kernelSS << "    data.func(" << sv_arg << ", 0, idxMax, data.mPtr);\n";

    if (config.installTimer) {
        kernelSS << "    tok = clock::now();\n"
                    << "    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count() << \" ms: \" << data.info << \"\\n\";\n"
                    << "    tic = clock::now();\n";
    }

    // meta data data type
    metaDataSS << "struct _meta_data_t_ {\n"
               << "  void (*func)(" << realTy << "*, " ;
    if (isSepKernel)     
        metaDataSS << realTy << "*, ";
    metaDataSS << "uint64_t, uint64_t, const void*);\n"
               << "  unsigned opCount;\n"
               << "  unsigned nqubits;\n";
    if (config.installTimer)
        metaDataSS << "  const char* info;\n";
    metaDataSS << "  const " << realTy << "* mPtr;\n"
               << "};\n"
               << "static const _meta_data_t_ _metaData[] = {\n";
    
    auto allBlocks = graph.getAllBlocks();
    if (forceInOrder)
        std::sort(allBlocks.begin(), allBlocks.end(),
                [](GateBlock* a, GateBlock* b) { return a->id < b->id; });

    for (const auto& block : allBlocks) {
        const auto& gate = *(block->quantumGate);
        std::string kernelName = "kernel_block_" + std::to_string(block->id);
        irGenerator.generateKernelDebug(gate, debugLevel, kernelName);
        // if (config.dumpIRToMultipleFiles) {
        //     std::error_code ec;
        //     llvm::raw_fd_ostream irFile(fileName + "_ir/" + kernelName + ".ll", ec);
        //     irGenerator.getModule().setModuleIdentifier(kernelName + "_module");
        //     irGenerator.getModule().setSourceFileName(kernelName + ".ll");
        //     irGenerator.getModule().print(irFile, nullptr);
        //     irFile.close();
        //     irGenerator.~IRGenerator();

        //     new (&irGenerator) IRGenerator(config.irConfig);
        // }

        externSS << " void " << kernelName << "("
                 << realTy << "*, ";
        if (isSepKernel)
            externSS << realTy << "*, ";
        externSS << "uint64_t, uint64_t, const void*);\n";
        
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
        const std::vector<std::complex<double>>&
            cdata = std::get<GateMatrix::c_matrix_t>(block->quantumGate->gateMatrix._matrix).data;
        for (const auto& elem : cdata)
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

    if (config.irConfig.precision == 32)
        hFile << "#define USING_F32\n";
    if (!isSepKernel)
        hFile << "#define USING_ALT_KERNEL\n";

    if (config.multiThreaded)
        hFile << "#include <vector>\n"
                 "#include <thread>\n"
                 "#define MULTI_THREAD_SIMULATION_KERNEL\n\n";

    hFile << "#include <cstdint>\n"
          << "#define DEFAULT_NQUBITS " << graph.nqubits << "\n"
          << "#define SIMD_S " << config.irConfig.simd_s << "\n"
          << externSS.str() << "\n"
          << metaDataSS.str() << "\n"
          << kernelSS.str() << "\n";

    hFile.close();

    // if (!config.dumpIRToMultipleFiles) {
        std::error_code ec;
        llvm::raw_fd_ostream irFile(fileName + ".ll", ec);
        irGenerator.getModule()->setModuleIdentifier(fileName + "_module");
        irGenerator.getModule()->setSourceFileName(fileName + ".ll");
        irGenerator.getModule()->print(irFile, nullptr);
        irFile.close();
    // }
}