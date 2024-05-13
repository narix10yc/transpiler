#include "simulation/ir_generator.h"
#include "llvm/Support/CommandLine.h"

using namespace simulation;
using namespace llvm;

int main(int argc, char** argv) {
    IRGenerator generator(1);
    generator.setRealTy(ir::RealTy::Double);
    generator.setAmpFormat(ir::AmpFormat::Separate);

    auto u2q = ir::U2qGate { ir::ComplexMatrix4{
        {2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2},
        {2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2}
    }, 2, 0};

    generator.genU2q(u2q);

    if (argc > 1) {
        std::error_code ec;
        raw_fd_ostream fIR(argv[1], ec);
        if (ec) {
            errs() << "Error opening file: " << ec.message() << "\n";
            return 1;
        }
        generator.getModule().print(fIR, nullptr);
    } else {
        generator.getModule().print(errs(), nullptr);
    }

    return 0;
}