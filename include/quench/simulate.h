#ifndef QUENCH_SIMULATE_H
#define QUENCH_SIMULATE_H

#include "quench/GateMatrix.h"
#include "quench/CircuitGraph.h"
#include <iomanip>

namespace quench::simulate {

inline size_t insertZeroBit(size_t number, int index) {
    size_t left, right;
    left = (number >> index) << index;
    right = number - left;
    return (left << 1) ^ right;
}

template<typename real_ty = double>
void applyGeneral(quench::cas::Complex<real_ty>* sv,
                  const quench::cas::GateMatrix& gate,
                  const std::vector<unsigned>& qubits,
                  unsigned nqubits)
{
    assert(gate.nqubits == qubits.size());
    using complex_t = quench::cas::Complex<real_ty>;

    std::cerr << "applyGeneral (nqubits = " << nqubits << ") on qubits ";
    for (const auto& q : qubits)
        std::cerr << q << " ";
    std::cerr << "\nwith gate\n";
    gate.print(std::cerr);

    const auto& K = gate.N;
    std::vector<complex_t> constMatrix(K*K);
    for (unsigned r = 0; r < K; r++) {
        for (unsigned c = 0; c < K; c++) {
            auto idx = r * K + c;
            auto reE = gate.matrix[idx].real.getExprValue();
            assert(reE.isConstant);
            auto imE = gate.matrix[idx].imag.getExprValue();
            assert(imE.isConstant);
            constMatrix[idx] = {reE.value, imE.value};
        }
    }

    // std::cerr << "constMatrix:\n";
    // for (size_t r = 0; r < K; r++) {
    //     for (size_t c = 0; c < K; c++) {
    //         const auto& elem = constMatrix[r * K + c];
    //         std::cerr << elem.real << " + " << elem.imag << "i, ";
    //     }
    //     std::cerr << "\n";
    // }

    std::vector<size_t> qubitsPower;
    for (const auto& q : qubits)
        qubitsPower.push_back(1 << q);

    auto qubitsSorted = qubits;
    std::sort(qubitsSorted.begin(), qubitsSorted.end());

    std::vector<size_t> idxVector(K);
    std::vector<complex_t> updatedAmp(K);

    for (size_t t = 0; t < (1 << (nqubits - gate.nqubits)); t++) {
        // extract indices
        for (size_t i = 0; i < K; i++) {
            size_t idx = t;
            for (const auto q : qubitsSorted)
                idx = insertZeroBit(idx, q);
            for (unsigned q = 0; q < gate.nqubits; q++) {
                if ((i & (1 << q)) > 0)
                    idx |= (1 << qubits[q]);
            }
            idxVector[i] = idx;
        }

        // std::cerr << "t = " << t << ": [";
        // for (const auto& idx : idxVector)
        //     std::cerr << std::bitset<4>(idx) << ",";
        // std::cerr << "]\n";

        // multiply
        for (size_t i = 0; i < K; i++) {
            updatedAmp[i] = { 0.0, 0.0 };
            for (size_t ii = 0; ii < K; ii++)
                updatedAmp[i] += sv[idxVector[ii]] * constMatrix[i*K + ii];
        }
        // store
        for (size_t i = 0; i < K; i++) {
            // std::cerr << "idx " << i << " new amp = " << updatedAmp[i].real 
            //           << " + " << updatedAmp[i].imag << "i"
            //           << " store back at position " << idxVector[i] << "\n";
            sv[idxVector[i]] = updatedAmp[i];
        }
    }

}

} // namespace quench::simulate
#endif // QUENCH_SIMULATE_H