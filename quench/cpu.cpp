#include "quench/cpu.h"
#include "simulation/ir_generator.h"
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace quench::cpu;
using IRGenerator = simulation::IRGenerator;
using CircuitGraph = quench::circuit_graph::CircuitGraph;

void CodeGeneratorCPU::generate(const CircuitGraph& graph) {
    IRGenerator irGenerator(config.s);
    irGenerator.setVerbose(0);
    const auto allBlocks = graph.getAllBlocks();

    std::stringstream externSS;
    std::stringstream matrixSS;
    std::stringstream kernelSS;

    externSS << "extern \"C\" {\n";
    matrixSS << "const static double _mPtr[] = {\n";
    kernelSS << "void simulation_kernel(double* re, double* im) {\n";

    unsigned matrixPosition = 0;

    for (const auto& block : allBlocks) {
        const auto gate = block->toQuantumGate();
        std::string kernelName = "kernel_block_" + std::to_string(block->id);
        irGenerator.generateKernel(gate, kernelName);

        externSS << " void " << kernelName << "(double*, double*, uint64_t, uint64_t, const void*);\n";

        matrixSS << " ";
        for (const auto& elem : gate.matrix.matrix.constantMatrix.data) {
            matrixSS << std::setprecision(16) << elem.real() << ","
                     << std::setprecision(16) << elem.imag() << ", ";
        }
        matrixSS << "\n";

        kernelSS << " " << kernelName << "(re, im, 0, "
                 << (1 << (graph.nqubits - gate.qubits.size() - config.s)) << ", "
                 << "_mPtr + " << matrixPosition << ");\n";
        matrixPosition += gate.matrix.matrix.getSize() * gate.matrix.matrix.getSize() * 2;
    }

    externSS << "};\n";
    matrixSS << "};\n";
    kernelSS << "};\n";

    std::ofstream hFile(config.fileName + ".h");
    assert(hFile.is_open());
    hFile << "#include <cstdint>\n\n"
          << externSS.str() << "\n"
          << matrixSS.str() << "\n"
          << kernelSS.str() << "\n";
    hFile.close();

    std::error_code ec;
    llvm::raw_fd_ostream irFile(config.fileName + ".ll", ec);
    irGenerator.getModule().print(irFile, nullptr);

    irFile.close();
}