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

    simulation::IRGenerator generator(3);
    generator.setVerbose(999);
    generator.vecSizeInBits = 4;

    auto m1 = GateMatrix::FromName("u3", {M_PI / 2, 0.0, M_PI});
    m1.matrix.constantMatrix = m1.matrix.constantMatrix.leftKronI();
    m1.nqubits += 1;
    m1.N *= 2;
    // m1.printMatrix(std::cerr) << "\n";

    auto m2 = m1.permute({1, 0});
    // m2.printMatrix(std::cerr);

    // QuantumGate gate1(m1, {8,4});
    // auto gate2 = gate1.lmatmul({m1, {4,6}});

    // QuantumGate gate1(m1, {8,4});
    // auto gate2 = gate1.lmatmul({m1, {4,0}});

    QuantumGate gate1(m1, {5,2});
    auto gate2 = gate1.lmatmul({m1, {1,2}});

    gate2.matrix.printMatrix(std::cerr) << "\n";

    generator.generateKernel(gate2, "funcName");
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