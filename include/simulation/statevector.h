#ifndef SIMULATION_STATEVECTOR_H_
#define SIMULATION_STATEVECTOR_H_

#include <cstdlib>
#include <random>

namespace simulation::sv {

template<typename real_ty>
class Statevector {
public:
    uint8_t nqubits;
    uint64_t N;
    real_ty* real;
    real_ty* imag;

    Statevector(uint8_t nqubits, bool rand=false) : nqubits(nqubits), N(1 << nqubits) {
        real = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        imag = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        if (rand)
            randomize();
        else
            real[0] = 1; // init to |0...0>     
    }

    ~Statevector() { free(real); free(imag); }

    Statevector(const Statevector&) = delete;
    Statevector& operator=(const Statevector& that) {
        if (this != &that) {
            for (size_t i = 0; i < N; i++) {
                real[i] = that.real[i];
                imag[i] = that.imag[i];
            }   
        }
        return *this;
    }
    Statevector(Statevector&&) = delete;
    Statevector& operator=(Statevector&&) = delete;

    double normSquared() const {
        double s = 0;
        for (size_t i = 0; i < N; i++) {
            s += real[i] * real[i];
            s += imag[i] * imag[i];
        } 
        return s;
    }

    double norm() const { return sqrt(normSquared()); }

    void normalize() {
        double n = norm();
        for (size_t i = 0; i < N; i++) {
            real[i] /= n;
            imag[i] /= n;
        } 
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<real_ty> d { 0, 1 };
        
        for (size_t i = 0; i < N; i++) {
            real[i] = d(gen);
            imag[i] = d(gen);
        }
        normalize();
    }
};

template<typename real_ty>
double fidelity(Statevector<real_ty> sv1, Statevector<real_ty> sv2) {
    double re, im;
    for (size_t i = 0; i < sv1.N; i++) {
        re += ( sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
        im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
    }
    return re * re + im * im;
}


} // namespace simulation::sv

#endif // SIMULATION_STATEVECTOR_H_