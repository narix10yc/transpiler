#ifndef SIMULATION_STATEVECTOR_H_
#define SIMULATION_STATEVECTOR_H_

#include <cstdlib>
#include <random>
#include <iostream>

namespace simulation::sv {

template<typename real_ty>
class StatevectorSep;

template<typename real_ty>
class StatevectorAlt;

template<typename real_ty>
class StatevectorSep {
public:
    unsigned nqubits;
    uint64_t N;
    real_ty* real;
    real_ty* imag;

    StatevectorSep(unsigned nqubits, bool initialize=false)
            : nqubits(nqubits), N(1 << nqubits) {
        real = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        imag = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        if (initialize) {
            for (size_t i = 0; i < (1 << nqubits); i++) {
                real[i] = 0;
                imag[i] = 0;
            }
            real[0] = 1.0;
        }
    }

    StatevectorSep(const StatevectorSep& that) : nqubits(that.nqubits), N(that.N) {
        real = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        imag = (real_ty*) aligned_alloc(64, N * sizeof(real_ty));
        for (size_t i = 0; i < that.N; i++) {
            real[i] = that.real[i];
            imag[i] = that.imag[i];
        }
    }

    StatevectorSep(StatevectorSep&&) = delete;

    ~StatevectorSep() { free(real); free(imag); }

    StatevectorSep& operator=(const StatevectorSep& that) {
        if (this != &that) {
            for (size_t i = 0; i < N; i++) {
                real[i] = that.real[i];
                imag[i] = that.imag[i];
            }   
        }
        return *this;
    }

    StatevectorSep& operator=(StatevectorSep&&) = delete;

    void copyValueFrom(const StatevectorAlt<real_ty>&);

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

    void print(std::ostream& os) const {
        // const char* red = "\033[31m";
        const char* cyan = "\033[36m";
        const char* bold = "\033[1m";
        const char* reset = "\033[0m";
        const auto print_number = [&](size_t idx) {
            if (real[idx] >= 0)
                os << " ";
            os << real[idx];
            if (imag[idx] >= 0)
                os << "+";
            os << imag[idx] << "i";
        };

        if (N > 32) {
            os << bold << cyan << "Warning: " << reset << "statevector has more "
                "than 5 qubits, only the first 32 entries are shown.\n";
        }
        for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
            os << i << ": ";
            print_number(i);
            os << "\n";
        }   
    }
};


template<typename real_ty>
class StatevectorAlt {
public:
    unsigned nqubits;
    uint64_t N;
    real_ty* data;

    StatevectorAlt(unsigned nqubits, bool initialize=false) 
            : nqubits(nqubits), N(1 << nqubits) {
        data = (real_ty*) aligned_alloc(64, 2 * N * sizeof(real_ty));
        if (initialize) {
            for (size_t i = 0; i < (1 << (nqubits+1)); i++)
                data[i] = 0;
            data[0] = 1;   
        }
    }

    StatevectorAlt(const StatevectorAlt& that) : nqubits(that.nqubits), N(that.N) {
        data = (real_ty*) aligned_alloc(64, 2 * N * sizeof(real_ty));
        for (size_t i = 0; i < 2 * that.N; i++)
            data[i] = that.data[i];
    }

    StatevectorAlt(StatevectorAlt&&) = delete;

    ~StatevectorAlt() { free(data); }

    StatevectorAlt& operator=(const StatevectorAlt& that) {
        if (this != &that) {
            for (size_t i = 0; i < 2*N; i++)
                data[i] = that.data[i];   
        }
        return *this;
    }

    StatevectorAlt& operator=(StatevectorAlt&&) = delete;

    void copyValueFrom(const StatevectorSep<real_ty>&);

    double normSquared() const {
        double s = 0;
        for (size_t i = 0; i < 2*N; i++)
            s += data[i] * data[i];
        return s;
    }

    double norm() const { return sqrt(normSquared()); }

    void normalize() {
        double n = norm();
        for (size_t i = 0; i < 2*N; i++)
            data[i] /= n;
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<real_ty> d { 0, 1 };
        for (size_t i = 0; i < 2*N; i++)
            data[i] = d(gen);
        normalize();
    }

    void print(std::ostream& os) const {
        // const char* red = "\033[31m";
        const char* cyan = "\033[36m";
        const char* bold = "\033[1m";
        const char* reset = "\033[0m";
        const auto print_number = [&](size_t idx) {
            if (data[2*idx] >= 0)
                os << " ";
            os << data[2*idx] << "+";
            if (data[2*idx+1] >= 0)
                os << " ";
            os << "i" << data[2*idx+1];
        };

        if (N > 32) {
            os << bold << cyan << "Warning: " << reset << "statevector has more "
                "than 5 qubits, only the first 32 entries are shown.\n";
        }
        for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
            os << i << ": ";
            print_number(i);
            os << "\n";
        }   
    }
};


template<typename real_ty>
double fidelity(const StatevectorSep<real_ty>& sv1, const StatevectorSep<real_ty>& sv2) {
    assert(sv1.nqubits == sv2.nqubits);

    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < sv1.N; i++) {
        re += ( sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
        im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
    }
    return re * re + im * im;
}

template<typename real_ty>
double fidelity(const StatevectorSep<real_ty>& sep, const StatevectorAlt<real_ty>& alt) {
    assert(sep.nqubits == alt.nqubits);

    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < sep.N; i++) {
        re += ( sep.real[i] * alt.data[2*i] + sep.imag[i] * alt.data[2*i+1]);
        im += (-sep.real[i] * alt.data[2*i+1] + sep.imag[i] * alt.data[2*i]);
    }
    return re * re + im * im;
}

template<typename real_ty>
double fidelity(const StatevectorAlt<real_ty>& alt, const StatevectorSep<real_ty>& sep) {
    return fidelity(sep, alt);
}


template<typename real_ty>
void StatevectorSep<real_ty>::copyValueFrom(const StatevectorAlt<real_ty>& alt) {
    assert(nqubits == alt.nqubits);

    for (size_t i = 0; i < N; i++) {
        real[i] = alt.data[2*i];
        imag[i] = alt.data[2*i+1];
    }
}

template<typename real_ty>
void StatevectorAlt<real_ty>::copyValueFrom(const StatevectorSep<real_ty>& sep) {
    assert(nqubits == sep.nqubits);
    
    for (size_t i = 0; i < N; i++) {
        data[2*i] = sep.real[i];
        data[2*i+1] = sep.imag[i];
    }
}

} // namespace simulation::sv

#endif // SIMULATION_STATEVECTOR_H_