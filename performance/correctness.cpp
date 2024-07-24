#include "gen_file.h"
#include "utils/iocolor.h"
#include "utils/statevector.h"
#include "timeit/timeit.h"
#include <iomanip>
#include <iostream>
#include <quench/simulate.h>

#ifdef USING_F32
    using real_t = float;
#else 
    using real_t = double;
#endif

using namespace timeit;
using namespace Color;
using namespace quench::simulate;
using namespace utils::statevector;
using namespace quench::quantum_gate;


int main(int argc, char** argv) {
    const int nqubits = 5;
    const std::vector<unsigned> targetQubits = { 0, 1 };

    auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
    auto gate = QuantumGate(mat, { targetQubits[0] });
    gate = gate.lmatmul({ mat , { targetQubits[1] }});    

    StatevectorComp<real_t> sv_c(nqubits);
    StatevectorSep<real_t> sv_s(nqubits);

    sv_c.randomize();
    
    for (unsigned i = 0; i < sv_c.N; i++) {
        // sv_c.data[i] = { 1.0, 1.0 };
        sv_s.real[i] = sv_c.data[i].real();
        sv_s.imag[i] = sv_c.data[i].imag();
    }

    applyGeneral(sv_c.data, gate.gateMatrix, gate.qubits, nqubits);
    sv_c.print(std::cerr);
    
    uint64_t idxMax = 1ULL << (sv_s.nqubits - S_VALUE - 2);
    // uint64_t idxMax = 1;
    _metaData[0].func(sv_s.real, sv_s.imag, 0, idxMax, _metaData[0].mPtr);

    sv_s.print(std::cerr);

    for (unsigned i = 0; i < sv_s.N; i++) {
        if (std::abs(sv_s.real[i] - sv_c.data[i].real()) > 1e-8)
            std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i << " real\n";
        
        if (std::abs(sv_s.imag[i] - sv_c.data[i].imag()) > 1e-8)
            std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i << " imag\n";
    }

    return 0;
}
