#include "simulation/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace simulation;

std::string getDefaultU2qFuncName(const ir::U2qGate& u2q, const IRGenerator& gen) {
    std::stringstream ss;
    ss << ((gen.realTy == ir::RealTy::Double) ? "f64" : "f32") << "_"
       << "s" << gen.vecSizeInBits << "_"
       << ((gen.ampFormat == ir::AmpFormat::Separate) ? "sep" : "alt") << "_"
       << "u2q";
    return ss.str();
}

Function* IRGenerator::genU2q(const ir::U2qGate& u2q, std::string funcName) {
    return nullptr;
}
