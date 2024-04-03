#include "simulation/cpu.h"


namespace {
void applyGateCZ
(double* real, double* imag, unsigned nqubits, unsigned q1, unsigned q2) {
    unsigned qSmall, qLarge;
    if (q1 < q2) {
        qSmall = q1; qLarge = q2;
    } else {
        qSmall = q2; qLarge = q1;
    }
    size_t k0 = (1 << qSmall) + (1 << qLarge);
    // for (size_t t = 0; t < (1 << nqubits); t += (1 << (qSmall + 1))) {
    //     for (size_t tt = 0; tt < count; tt++)
    //     {
    //         /* code */
    //     }
        
    // }
    
}
} // <anonymous> namespace

using namespace simulation;
using namespace openqasm::ast;

void CPUGenContext::generate(const RootNode& root) {
    root.genCPU(*this);
}

// void GateApplyStmt::genCPU(const CPUGenContext& ctx) const {
    // std::cerr << "Gate Apply\n";
// }


int main() {
    return 0;
}