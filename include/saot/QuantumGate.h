#ifndef SAOT_QUANTUMGATE_H
#define SAOT_QUANTUMGATE_H

#include <vector>
#include <iostream>
#include <complex>
#include "quench/Polynomial.h"

namespace saot::quantum_gate {

template<typename T>
class QubitMatrix {
public:
    int nqubits;
    std::vector<T> data;

    QubitMatrix(int nqubits) : nqubits(nqubits), data(1 << (nqubits + 1)) {}
};

class QuantumGate {
public:
    std::vector<int> qubits;
    QubitMatrix<std::complex<double>> coefMatrix;
    QubitMatrix<quench::cas::Polynomial> polynomialMatrix;

};

} // namespace saot::quantum_gate

#endif // SAOT_QUANTUMGATE_H