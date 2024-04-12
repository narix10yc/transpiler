#include "simulation/irGen.h"
#include "llvm/Support/CommandLine.h"
#include <sstream>

using namespace llvm;
using namespace simulation;

int main(int argc, char **argv) {
    using RealTy = simulation::IRGenerator::RealTy;

    cl::opt<std::string> 
    FileName("F", cl::Prefix, cl::desc("file name"), cl::init(""));

    cl::opt<unsigned>
    Qubit("K", cl::Prefix, cl::desc("act on which qubit"), cl::Required);

    cl::opt<std::string>
    Ty("type", cl::desc("float or double"), cl::init("double"));

    cl::opt<std::string>
    Theta("theta", cl::desc("theta value"), cl::init("none"));

    cl::opt<std::string>
    Phi("phi", cl::desc("phi value"), cl::init("none"));

    cl::opt<std::string>
    Lambda("lambda", cl::desc("lambda value"), cl::init("none"));

    cl::opt<double>
    Thres("thres", cl::desc("threshold"), cl::init(1e-8));

    cl::opt<unsigned>
    VecSize("S", cl::Prefix, cl::desc("vector size"), cl::init(2));

    cl::opt<bool>
    InUnitsOfPI("in-units-of-pi", cl::desc("is the input angles in the unit of pi"),
        cl::init("False"));

    cl::ParseCommandLineOptions(argc, argv, "");

    IRGenerator gen(VecSize);

    RealTy ty = RealTy::Double;
    if (Ty == "double")
        ty = RealTy::Double;
    else if (Ty == "float")
        ty = RealTy::Float;
    else {
        errs() << "Unrecognized type " << ty << "\n";
        return 1;
    }

    double _multiple = (InUnitsOfPI) ? 3.14159265358979323846 : 1.0;
    std::optional<double> theta, phi, lambd;
    try {
        if (Theta != "none")
            theta = std::stod(Theta) * _multiple;
        if (Phi != "none")
            phi = std::stod(Phi) * _multiple;
        if (Lambda != "none")
            lambd = std::stod(Lambda) * _multiple;
    } catch (...) {
        errs() << "Unable to process input angles\n";
        return 1;
    }

    std::stringstream ss;
    ss << "kernel_" << ((ty == RealTy::Double) ? "f64" : "f32")
       << "_K" << Qubit << "_S" << VecSize;
    std::string funcName = ss.str();
    
    auto u3 = U3Gate::FromAngles(static_cast<uint8_t>(Qubit), theta, phi, lambd);

    gen.genU3(u3, funcName, ty);

    // print to file 
    if (FileName != "") {
        std::error_code ec;
        raw_fd_ostream fIR(FileName, ec);
        if (ec) {
            errs() << "Error opening file: " << ec.message() << "\n";
            return 1;
        }
        gen.getModule().print(fIR, nullptr);
    } else {
        gen.getModule().print(errs(), nullptr);
    }

    return 0;
}