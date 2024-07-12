#include "quench/cpu.h"
#include "simulation/ir_generator.h"
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace quench::cpu;
using IRGenerator = simulation::IRGenerator;
using CircuitGraph = quench::circuit_graph::CircuitGraph;

void CodeGeneratorCPU::generate(const CircuitGraph& graph, int verbose) {
    IRGenerator irGenerator(config.s);
    irGenerator.setVerbose(verbose);
    const auto allBlocks = graph.getAllBlocks();

    std::stringstream externSS;
    std::stringstream matrixSS;
    std::stringstream kernelSS;

    externSS << "extern \"C\" {\n";
    matrixSS << "const static double _mPtr[] = {\n";

    if (config.nthreads > 1)
        kernelSS << "void simulation_kernel(double* re, double* im, "
                    "const int nthreads = " << config.nthreads << ") {\n";
    else
        kernelSS << "void simulation_kernel(double* re, double* im) {\n";

    if (config.nthreads > 1)
        kernelSS << " std::array<std::thread, " << config.nthreads << "> threads;\n"
                 << " size_t chunkSize;\n";

    if (config.installTimer)
        kernelSS << "using clock = std::chrono::high_resolution_clock;\n"
                    "auto tic = clock::now();\n"
                    "auto tok = clock::now();\n";

    unsigned matrixPosition = 0;
    for (const auto& block : allBlocks) {
        const auto& gate = *(block->quantumGate);
        std::string kernelName = "kernel_block_" + std::to_string(block->id);
        irGenerator.generateKernel(gate, kernelName);

        externSS << " void " << kernelName << "(double*, double*, uint64_t, uint64_t, const void*);\n";

        matrixSS << " ";
        for (const auto& elem : gate.gateMatrix.matrix.constantMatrix.data) {
            matrixSS << std::setprecision(16) << elem.real() << ","
                     << std::setprecision(16) << elem.imag() << ", ";
        }
        matrixSS << "\n";

        size_t idxMax = (1 << (graph.nqubits - gate.qubits.size() - config.s));
        if (config.nthreads > 1) {
            size_t chunkSize = idxMax / config.nthreads;
            assert(chunkSize * config.nthreads == idxMax
                   && "Uneven split not implemented yet...");
                   
            kernelSS << " chunkSize = " << idxMax << "ULL / nthreads;\n"

                     << " for (unsigned i = 0; i < nthreads; i++)\n"
                     << "  threads[i] = std::thread(" << kernelName << ", re, im, "
                     << "i*chunkSize, (i+1)*chunkSize, "
                     << "_mPtr + " << matrixPosition << ");\n"
            
                     << " for (unsigned i = 0; i < nthreads; i++)\n"
                     << "  threads[i].join();\n";
        }
        else 
            kernelSS << " " << kernelName << "(re, im, 0, "
                     << idxMax << "ULL, " << "_mPtr + " << matrixPosition << ");\n";
        
        if (config.installTimer)
            kernelSS << " PRINT_BLOCK_TIME(" << block->id << ")\n";

        auto matrixSize = gate.gateMatrix.matrix.getSize();
        matrixPosition += 2 * matrixSize * matrixSize;
    }
    
    externSS << "};\n";
    matrixSS << "};\n";
    kernelSS << "};\n";

    std::ofstream hFile(config.fileName + ".h");
    assert(hFile.is_open());

    if (config.installTimer)
        hFile << "#include <chrono>\n"
                 "#include <iostream>\n"
                 
                 "#define PRINT_BLOCK_TIME(BLOCK)\\\n"
                 "  tok = clock::now();\\\n"
                 "  std::cerr << \" Block \" << BLOCK << \" takes \" << "
                 "std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count() << \" ms;\\n\";\\\n"
                 "  tic = clock::now();\n\n";

    if (config.nthreads > 1)
        hFile << "#include <array>\n"
                 "#include <thread>\n"
                 "#define MULTI_THREAD_SIMULATION_KERNEL\n\n";

    hFile << "#include <cstdint>\n"
          << externSS.str() << "\n"
          << matrixSS.str() << "\n"
          << kernelSS.str() << "\n";
    hFile.close();

    std::error_code ec;
    llvm::raw_fd_ostream irFile(config.fileName + ".ll", ec);
    irGenerator.getModule().print(irFile, nullptr);

    irFile.close();
}