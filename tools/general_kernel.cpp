#include "simulation/ir_generator.h"
#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"
#include <sstream>

#ifndef M_PI
    #define M_PI (3.14159265358979323846264338327950288)
#endif

using namespace Color;
using namespace quench::quantum_gate;

int main(int argc, char** argv) {
    simulation::IRGenerator generator;
    generator.setVerbose(999);
    generator.vecSizeInBits = 1;

    auto matrix_u1q = GateMatrix::FromName("u3", {0.1, 0.2, 0.3});

    // auto matrix_h = GateMatrix::FromName("u3", {0.1, 0.2, 0.3});
    auto matrix_h = GateMatrix::FromName("h");

    QuantumGate gate = QuantumGate(matrix_h, 1);
                        // .lmatmul({matrix_h, 2});
                        // .lmatmul({matrix_h, 3});

    gate.displayInfo(std::cerr);
    
    generator.generateKernel(gate, "simulation_kernel_1");
    // generator.generateKernel(gate2, "funcName");
    if (argc > 1) {
        std::error_code ec;
        llvm::raw_fd_ostream fIR(argv[1], ec);
        if (ec) {
            llvm::errs() << "Error opening file: " << ec.message() << "\n";
            return 1;
        }
        generator.getModule().print(fIR, nullptr);
    } else {
        generator.getModule().print(llvm::errs(), nullptr);
    }


}