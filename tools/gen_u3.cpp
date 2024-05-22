#include "simulation/ir_generator.h"
#include "llvm/Support/CommandLine.h"
#include <sstream>

using namespace llvm;
using namespace simulation;

using RealTy = ir::RealTy;
using AmpFormat = ir::AmpFormat;

int main(int argc, char **argv) {
    cl::opt<std::string> 
    FileName("F", cl::Prefix, 
             cl::desc("file name to write to. If left empty, use std::cerr"),
             cl::init(""));

    cl::opt<unsigned>
    Qubit("K", cl::Prefix, cl::desc("act on which qubit"), cl::Required);

    cl::opt<std::string>
    Ty("type", cl::desc("float (f32) or double (f64)"), cl::init("f64"));

    cl::opt<std::string>
    Theta("theta", cl::desc("theta value, default 'none'"), cl::init("none"));

    cl::opt<std::string>
    Phi("phi", cl::desc("phi value, default 'none'"), cl::init("none"));

    cl::opt<std::string>
    Lambda("lambda", cl::desc("lambda value, default 'none'"), cl::init("none"));

    cl::opt<double>
    Thres("thres", cl::desc("threshold, default 1e-8"), cl::init(1e-8));

    cl::opt<unsigned>
    VecSize("S", cl::Prefix, cl::desc("vector size, default 2"), cl::init(2));

    cl::opt<bool>
    InUnitsOfPI("in-units-of-pi", cl::desc("is the input angles in the unit of pi"),
        cl::init("False"));
    
    cl::opt<bool>
    AlternatingFormat("alt", cl::desc("use alternating format"), cl::init("false"));

    cl::ParseCommandLineOptions(argc, argv, "");

    IRGenerator generator(VecSize);

    RealTy ty = RealTy::Double;
    if (Ty == "double" || Ty == "f64")
        ty = RealTy::Double;
    else if (Ty == "float" || Ty == "f32")
        ty = RealTy::Float;
    else {
        errs() << "Unrecognized type " << Ty << "\n";
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

    auto mat = ir::ComplexMatrix2::FromEulerAngles(theta, phi, lambd);
    auto u3 = ir::U3Gate { static_cast<uint8_t>(Qubit), mat };

    generator.setRealTy(ty);

    generator.setAmpFormat((AlternatingFormat) ? ir::AmpFormat::Alternating
                                               : ir::AmpFormat::Separate);

    generator.genU3(u3);

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