#ifndef UTILS_STATEVECTOR_H
#define UTILS_STATEVECTOR_H

#include <cstdlib>
#include <cassert>
#include <complex>
#include <random>
#include <iostream>
#include <bitset>

#include "utils/iocolor.h"
#include "utils/utils.h"

namespace utils::statevector {

template<typename real_t>
class StatevectorSep;

template<typename real_t>
class StatevectorAlt;

template<typename real_t>
class StatevectorSep {
public:
    unsigned nqubits;
    uint64_t N;
    real_t* real;
    real_t* imag;

    StatevectorSep(int nqubits, bool initialize=false)
            : nqubits(static_cast<unsigned>(nqubits)), N(1 << nqubits) {
        assert(nqubits > 0);
        real = (real_t*) aligned_alloc(64, N * sizeof(real_t));
        imag = (real_t*) aligned_alloc(64, N * sizeof(real_t));
        if (initialize) {
            for (size_t i = 0; i < (1 << nqubits); i++) {
                real[i] = 0;
                imag[i] = 0;
            }
            real[0] = 1.0;
        }
    }

    StatevectorSep(const StatevectorSep& that) : nqubits(that.nqubits), N(that.N) {
        real = (real_t*) aligned_alloc(64, N * sizeof(real_t));
        imag = (real_t*) aligned_alloc(64, N * sizeof(real_t));
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

    void copyValueFrom(const StatevectorAlt<real_t>&);

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
        std::normal_distribution<real_t> d { 0, 1 };
        
        for (size_t i = 0; i < N; i++) {
            real[i] = d(gen);
            imag[i] = d(gen);
        }
        normalize();
    }

    std::ostream& print(std::ostream& os) const {
        if (N > 32) {
            os << Color::BOLD << Color::CYAN_FG << "Warning: " << Color::RESET
               << "statevector has more than 5 qubits, "
                    "only the first 32 entries are shown.\n";
        }
        for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
            os << i << ": ";
            print_complex(os, {real[i], imag[i]});
            os << "\n";
        }
        return os;
    }
};

template<typename real_t>
class StatevectorAlt {
public:
    unsigned nqubits;
    uint64_t N;
    real_t* data;

    StatevectorAlt(unsigned nqubits, bool initialize=false) 
            : nqubits(nqubits), N(1 << nqubits) {
        data = (real_t*) aligned_alloc(64, 2 * N * sizeof(real_t));
        if (initialize) {
            for (size_t i = 0; i < (1 << (nqubits+1)); i++)
                data[i] = 0;
            data[0] = 1;   
        }
    }

    StatevectorAlt(const StatevectorAlt& that) : nqubits(that.nqubits), N(that.N) {
        data = (real_t*) aligned_alloc(64, 2 * N * sizeof(real_t));
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

    void copyValueFrom(const StatevectorSep<real_t>&);

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
        std::normal_distribution<real_t> d { 0, 1 };
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

template<typename real_t>
class StatevectorComp {
    using complex_t = std::complex<real_t>;
public:
    unsigned nqubits;
    size_t N;
    complex_t* data;
    StatevectorComp(unsigned nqubits) : nqubits(nqubits), N(1<<nqubits) {
        data = new complex_t[N];
    }

    ~StatevectorComp() { delete[] data; }

    StatevectorComp(StatevectorComp&&) = delete;
    StatevectorComp& operator=(StatevectorComp&&) = delete;

    StatevectorComp(const StatevectorComp& that)
        : nqubits(that.nqubits), N(that.N)
    {
        data = new complex_t[N];
        for (size_t i = 0; i < N; i++)
            data[i] = that.data[i];
    }

    StatevectorComp& operator=(const StatevectorComp& that) {
        assert(nqubits == that.nqubits);
        if (this == &that)
            return *this;

        for (size_t i = 0; i < N; i++)
            data[i] = that.data[i];
        return *this;
    }

    double normSquared() const {
        double sum = 0;
        for (size_t i = 0; i < N; i++) {
            sum += data[i].real() * data[i].real();
            sum += data[i].imag() * data[i].imag();
        }
        return sum;
    }

    double norm() const { return std::sqrt(normSquared()); }

    void normalize() {
        double n = norm();
        for (size_t i = 0; i < N; i++)
            data[i] /= n;
    }

    void zeroState() {
        for (size_t i = 0; i < N; i++)
            data[i] = {0.0, 0.0};
        data[0] = {1.0, 0.0};
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<real_t> d { 0, 1 };
        for (size_t i = 0; i < N; i++)
            data[i] = { d(gen), d(gen) };
        normalize();
    }

    std::ostream& print(std::ostream& os) const {
        using namespace Color;

        if (N > 32) {
            os << BOLD << CYAN_FG << "Warning: " << RESET << "statevector has more "
                "than 5 qubits, only the first 32 entries are shown.\n";
        }
        for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
            os << std::bitset<5>(i) << ": ";
            print_complex(os, data[i]);
            os << "\n";
        }
        return os;
    }
};


template<typename real_t>
static double fidelity(const StatevectorSep<real_t>& sv1, const StatevectorSep<real_t>& sv2) {
    assert(sv1.nqubits == sv2.nqubits);

    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < sv1.N; i++) {
        re += ( sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
        im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
    }
    return re * re + im * im;
}

template<typename real_t>
static double fidelity(const StatevectorSep<real_t>& sep, const StatevectorAlt<real_t>& alt) {
    assert(sep.nqubits == alt.nqubits);

    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < sep.N; i++) {
        re += ( sep.real[i] * alt.data[2*i] + sep.imag[i] * alt.data[2*i+1]);
        im += (-sep.real[i] * alt.data[2*i+1] + sep.imag[i] * alt.data[2*i]);
    }
    return re * re + im * im;
}

template<typename real_t>
double fidelity(const StatevectorAlt<real_t>& alt, const StatevectorSep<real_t>& sep) {
    return fidelity(sep, alt);
}


template<typename real_t>
void StatevectorSep<real_t>::copyValueFrom(const StatevectorAlt<real_t>& alt) {
    assert(nqubits == alt.nqubits);

    for (size_t i = 0; i < N; i++) {
        real[i] = alt.data[2*i];
        imag[i] = alt.data[2*i+1];
    }
}

template<typename real_t>
void StatevectorAlt<real_t>::copyValueFrom(const StatevectorSep<real_t>& sep) {
    assert(nqubits == sep.nqubits);
    
    for (size_t i = 0; i < N; i++) {
        data[2*i] = sep.real[i];
        data[2*i+1] = sep.imag[i];
    }
}

} // namespace utils::statevector

#endif // UTILS_STATEVECTOR_H