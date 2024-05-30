#include "simulation/ir_generator.h"
#include "llvm/Support/CommandLine.h"
#include <sstream>
#include <filesystem>

using namespace llvm;
using namespace simulation;

using RealTy = ir::RealTy;

int main(int argc, char **argv) {
    cl::opt<std::string> 
    FileName("F", cl::Prefix, cl::desc("file name"), cl::init(""));

    cl::opt<unsigned>
    QubitK("K", cl::Prefix, cl::desc("act on which qubit"), cl::Required);

    cl::opt<unsigned>
    QubitL("L", cl::Prefix, cl::desc("act on which qubit"), cl::Required);

    cl::opt<unsigned>
    VecSize("S", cl::Prefix, cl::desc("vector size"), cl::init(2));

    cl::opt<std::string>
    Ty("type", cl::desc("float (f32) or double (f64)"), cl::init("f64"));

    cl::opt<std::string>
    MatrixID("matrix", cl::desc("matrix"), cl::init("FFFFFFFFFFFFFFFF"));

    cl::opt<double>
    Thres("thres", cl::desc("threshold"), cl::init(1e-8));

    cl::ParseCommandLineOptions(argc, argv, "");

    if (QubitK < QubitL) {
        errs() << "K has to be larger than L\n";
        return 1;
    }

    IRGenerator generator(VecSize);
    if (std::filesystem::exists(FileName.c_str()))
        generator.loadFromFile(FileName);

    RealTy ty = RealTy::Double;
    if (Ty == "double" || Ty == "f64")
        ty = RealTy::Double;
    else if (Ty == "float" || Ty == "f32")
        ty = RealTy::Float;
    else {
        errs() << "Unrecognized type " << Ty << "\n";
        return 1;
    }

    uint64_t matrixID = std::stoull(MatrixID, nullptr, 16);

    ir::U2qGate u2q {static_cast<uint8_t>(QubitK), static_cast<uint8_t>(QubitL), matrixID};

    generator.setRealTy(ty);

    generator.genU2q(u2q);

    // print to file 
    if (FileName != "") {
        std::error_code ec;
        raw_fd_ostream fIR(FileName, ec);
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