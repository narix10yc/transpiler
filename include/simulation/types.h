#ifndef SIMULATION_TYPES_H_
#define SIMULATION_TYPES_H_

#include <optional>
#include <cmath>
#include <initializer_list>
#include <array>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <cassert>

namespace simulation {

template<typename real_t=double>
class Complex {
public:
    real_t real, imag;
    Complex() : real(0), imag(0) {}
    Complex(real_t real, real_t imag) : real(real), imag(imag) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imzg);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    Complex& operator*=(const Complex& other) {
        real_t r = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = r;
        return *this;
    }
};


template<typename real_t=double>
class SquareComplexMatrix {
    size_t size;
public:
    std::vector<Complex<real_t>> data;

    SquareComplexMatrix(size_t size) : size(size), data(size * size) {}
    SquareComplexMatrix(size_t size, std::initializer_list<Complex<real_t>> data)
        : size(size), data(data) {}

    size_t getSize() const { return size; }

    static SquareComplexMatrix Identity(size_t size) {
        SquareComplexMatrix m;
        for (size_t r = 0; r < size; r++)
            m.data[r*size + r].real = 1;
        return m;
    }

    SquareComplexMatrix matmul(const SquareComplexMatrix& other) {
        assert(size == other.size);

        SquareComplexMatrix m(size);
        for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
        for (size_t k = 0; k < size; k++) {
            // C_{ij} = A_{ik} B_{kj}
            m.data[i*size + j] += data[i*size + k] * other.data[k*size + j];
        } } }
        return m;
    }

    SquareComplexMatrix kron(const SquareComplexMatrix& other) const {
        size_t lsize = size;
        size_t rsize = other.size;
        size_t msize = lsize * rsize;
        SquareComplexMatrix m(msize);
        for (size_t lr = 0; lr < lsize; lr++) {
        for (size_t lc = 0; lc < lsize; lc++) {
        for (size_t rr = 0; rr < rsize; rr++) {
        for (size_t rc = 0; rc < rsize; rc++) {
            size_t r = lr * rsize + rr;
            size_t c = lc * rsize + rc;
            m.data[r*msize + c] = data[lr*lsize + lc] * other.data[rr*rsize + rc];
        } } } }
        return m;
    }

    SquareComplexMatrix leftKronI() const {
        SquareComplexMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(i*size + r) * size * size + (i*size + c)] = data[r*size + c];
        } } }
        return m;
    }

    SquareComplexMatrix rightKronI() const {
        SquareComplexMatrix m(size * size);
        for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < size; r++) {
        for (size_t c = 0; c < size; c++) {
            m.data[(r*size + i) * size * size + (c*size + i)] = data[r*size + c];
        } } }
        return m;
    }

    SquareComplexMatrix swapTargetQubits() const {
        assert(size == 4);

        return {4, {data[ 0], data[ 2], data[ 1], data[ 3],
                    data[ 8], data[10], data[ 9], data[11],
                    data[ 4], data[ 6], data[ 5], data[ 7],
                    data[12], data[14], data[13], data[15]}};
    }

    void print(std::ostream& os) const {
        for (size_t r = 0; r < size; r++) {
            for (size_t c = 0; c < size; c++) {
                auto re = data[r*size + c].real;
                auto im = data[r*size + c].imag;
                if (re >= 0)
                    os << " ";
                os << re;
                if (im >= 0)
                    os << "+";
                os << im << "i, ";
            }
            os << "\n";
        }
    }
};


namespace ir {
    enum class RealTy {
    Double, Float
};

enum class AmpFormat {
    Separate, Alternating
};

/// @brief Special complex matrix representation used in IR generation.
/// Its entries are ints, only with special meanings when equal to +1, -1, or 0.
/// All other cases are treated equal.
class ComplexMatrix2 {
public:
    std::array<int, 4> real, imag;
    ComplexMatrix2() = delete;
    explicit ComplexMatrix2(std::array<int, 4> real, std::array<int, 4> imag)
        : real(real), imag(imag) {}

    static ComplexMatrix2 
    FromEulerAngles(std::optional<double> theta, std::optional<double> phi, 
                    std::optional<double> lambd, double thres=1e-8);
};

class U3Gate {
public:
    uint8_t k;
    ComplexMatrix2 mat;
    U3Gate(uint8_t k, ComplexMatrix2 mat) : k(k), mat(mat) {}

    U3Gate(uint8_t k,
           std::optional<double> theta, std::optional<double> phi, 
           std::optional<double> lambd, double thres=1e-8)
        : k(k), mat(ComplexMatrix2::FromEulerAngles(theta, phi, lambd, thres)) {}

    static U3Gate FromID(uint32_t id);
    uint32_t getID() const;
    std::string getRepr() const;
};

class U2qGate {
public:
    uint8_t qLarge, qSmall;
    uint64_t mat;

    explicit U2qGate(uint8_t qLarge, uint8_t qSmall, uint64_t mat)
            : qLarge(qLarge), qSmall(qSmall), mat(mat) { 
        assert(qLarge > qSmall);
    }
    
    std::string getRepr() const;
};

} // namespace ir


class OptionalComplexMatrix2 {
public:
    std::optional<double> ar, br, cr, dr, ai, bi, ci, di;
public:
    OptionalComplexMatrix2(std::optional<double> ar, 
    std::optional<double> br, std::optional<double> cr,
    std::optional<double> dr, std::optional<double> bi,
    std::optional<double> ci, std::optional<double> di) 
    : ar(ar), br(br), cr(cr), dr(dr), ai({}), bi(bi), ci(ci), di(di) {}

    ir::ComplexMatrix2 ToIRMatrix(double thres=1e-8) const;
    
    static OptionalComplexMatrix2 
    FromEulerAngles(std::optional<double> theta,
                    std::optional<double> phi, 
                    std::optional<double> lambd);

    std::array<double, 8> toArray() const {
        return {
         ar.value_or(0.0), br.value_or(0.0), cr.value_or(0.0), dr.value_or(0.0), 
         ai.value_or(0.0), bi.value_or(0.0), ci.value_or(0.0), di.value_or(0.0)}; 
    }
    
}; // OptionalComplexMatrix2

template<typename real_t=double>
class ComplexMatrix2;

template<typename real_t=double>
class ComplexMatrix4;


template<typename real_t>
class ComplexMatrix2 {
public:
    std::array<real_t, 4> real, imag;

    ComplexMatrix2() : real(), imag() {}
    
    ComplexMatrix2(std::array<real_t, 4> real, std::array<real_t, 4> imag)
        : real(real), imag(imag) {}


    ir::ComplexMatrix2 ToIRMatrix(real_t threshold=1e-8) const {
        std::array<int, 4> realArr, imagArr;
        for (size_t i = 0; i < 4; i++) {
            if (abs(real[i] - 1) < threshold)
                realArr[i] = 1;
            else if (abs(real[i] + 1) < threshold)
                realArr[i] = -1;
            else if (abs(real[i]) < threshold)
                realArr[i] = 0;
            else
                realArr[i] = 2;

            if (abs(imag[i] - 1) < threshold)
                imagArr[i] = 1;
            else if (abs(imag[i] + 1) < threshold)
                imagArr[i] = -1;
            else if (abs(imag[i]) < threshold)
                imagArr[i] = 0;
            else
                imagArr[i] = 2;    
        }
        return ir::ComplexMatrix2 { realArr, imagArr };
    }

    static ComplexMatrix2
    FromEulerAngles(double theta, double phi, double lambd) {
        return {
            {
                cos(theta/2), -cos(lambd) * sin(theta/2),
                cos(phi) * sin(theta/2), cos(phi + lambd) * cos(theta/2)
            },
            {
                0, -sin(lambd) * sin(theta/2),
                sin(phi) * sin(theta/2), sin(phi + lambd) * cos(theta/2)
            }
        };
    }

    static ComplexMatrix2 Identity() {
        return {{1,0,0,1}, {}};
    }

    static ComplexMatrix2 Random() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<real_t> d { 0, 1 };
        ComplexMatrix2 mat {};
        for (size_t i = 0; i < 4; i++) {
            mat.real[i] = d(gen);
            mat.imag[i] = d(gen);
        }
        mat.imag[0] = 0.0;
        return mat;
    }

    /// @brief I otimes U
    ComplexMatrix4<real_t> leftKronI() const {
        return {
            {
                real[0], real[1], 0, 0,
                real[2], real[3], 0, 0,
                0, 0, real[0], real[1],
                0, 0, real[2], real[3]
            },
            {
                imag[0], imag[1], 0, 0,
                imag[2], imag[3], 0, 0,
                0, 0, imag[0], imag[1],
                0, 0, imag[2], imag[3]
            }
        };
    }

    /// @brief U otimes I 
    ComplexMatrix4<real_t> rightKronI() const {
        return {
            {
                real[0],      0, real[1],      0,
                      0, real[0],      0, real[1],
                real[2],      0, real[3],      0,
                      0, real[2],      0, real[3],
            },
            {
                imag[0],      0, imag[1],      0,
                      0, imag[0],      0, imag[1],
                imag[2],      0, imag[3],      0,
                      0, imag[2],      0, imag[3],
            }
        };
    }
};

template<typename real_t>
class ComplexMatrix4 {
public:
    std::array<real_t, 16> real, imag;

    ComplexMatrix4() : real(), imag() {}

    ComplexMatrix4(std::array<real_t, 16> real, std::array<real_t, 16> imag)
        : real(real), imag(imag) {}

    uint64_t toIRMatrix(real_t threshold=1e-8) const {
        uint64_t id = 0ULL;
        for (size_t i = 0; i < 16; i++) {
            if (abs(real[i]) < threshold) {} // add 0
            else if (abs(real[i] - 1) < threshold)
                id += 1ULL << (2*i);
            else if (abs(real[i] + 1) < threshold)
                id += 2ULL << (2*i);
            else
                id += 3ULL << (2*i);
            
            if (abs(imag[i]) < threshold) {} // add 0
            else if (abs(imag[i] - 1) < threshold)
                id += 1ULL << (2*i + 32);
            else if (abs(imag[i] + 1) < threshold)
                id += 2ULL << (2*i + 32);
            else
                id += 3ULL << (2*i + 32);
        }
        return id;
    }

    ComplexMatrix4 matmul(const ComplexMatrix4& other) {
        ComplexMatrix4 newMat;
        for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 4; k++) {
            // A_ij B_jk
            newMat.real[4*i+k] += real[4*i+j] * other.real[4*j+k]
                                 -imag[4*i+j] * other.imag[4*j+k];
            newMat.imag[4*i+k] += real[4*i+j] * other.imag[4*j+k]
                                 +imag[4*i+j] * other.real[4*j+k];
        }}}
        return newMat;
    }

    void print(std::ostream& os) const {
        const auto outNumber = [&](size_t idx) {
            if (real[idx] >= 0)
                os << " ";
            os << real[idx] << "+";
            if (imag[idx] >= 0)
                os << " ";
            os << imag[idx] << " i";
        };
        os << std::scientific << std::setprecision(4) << "[";
        for (size_t r = 0; r < 4; r++) {
            if (r > 0)
                os << " ";
            for (size_t c = 0; c < 4; c++) {
                outNumber(4*r + c);
                if (r == 3 && c == 3)
                    os << "]";
                else if (c == 3)
                    os << ",\n";
                else 
                    os << ", ";
            }
        }
    }
    
    static ComplexMatrix4 Identity() {
        return {{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, {}};
    }

    static ComplexMatrix4 Random() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<> d { 0, 1 };
        ComplexMatrix4 mat {};
        for (size_t i = 0; i < 16; i++) {
            mat.real[i] = d(gen);
            mat.imag[i] = d(gen);
        }
        return mat;
    }
    
    ComplexMatrix4 swapTargetQubits() const {
        return {
            {
                real[ 0], real[ 2], real[ 1], real[ 3],
                real[ 8], real[10], real[ 9], real[11],
                real[ 4], real[ 6], real[ 5], real[ 7],
                real[12], real[14], real[13], real[15]
            },
            {
                imag[ 0], imag[ 2], imag[ 1], imag[ 3],
                imag[ 8], imag[10], imag[ 9], imag[11],
                imag[ 4], imag[ 6], imag[ 5], imag[ 7],
                imag[12], imag[14], imag[13], imag[15]
            }
        };
    }
}; // ComplexMatrix4


class U3Gate {
public:
    ComplexMatrix2<> mat;
    uint8_t k;
public:
    U3Gate(ComplexMatrix2<> mat, uint8_t k) : mat(mat), k(k) {}

    ir::U3Gate ToIRGate() const {
        return ir::U3Gate { k, mat.ToIRMatrix() };
    }
};

class U2qGate {
public:
    uint8_t k, l;
    ComplexMatrix4<> mat;
public:
    U2qGate(uint8_t k, uint8_t l, const ComplexMatrix4<>& mat)
        : k(k), l(l), mat(mat) {}

    /// @brief Swap the target qubits 'in-place'. So k becomes l and l becomes 
    /// k. The matrix will change correspondingly.
    void swapTargetQubits() {
        uint8_t tmp = k; k = l; l = tmp;
        const auto& r = mat.real;
        const auto& i = mat.imag;
        mat = {
            {r[ 0], r[ 2], r[ 1], r[ 3],
             r[ 8], r[10], r[ 9], r[11],
             r[ 4], r[ 6], r[ 5], r[ 7],
             r[12], r[14], r[13], r[15]},
            {i[ 0], i[ 2], i[ 1], i[ 3],
             i[ 8], i[10], i[ 9], i[11],
             i[ 4], i[ 6], i[ 5], i[ 7],
             i[12], i[14], i[13], i[15]}
        };
    }

    ir::U2qGate ToIRGate() const {
        // convention: l is the less significant qubit
        if (l < k)
            return ir::U2qGate { k, l, mat.toIRMatrix() };
        
        auto u2q = *this;
        u2q.swapTargetQubits();
        return ir::U2qGate { u2q.k, u2q.l, u2q.mat.toIRMatrix() };
    }
    
};






} // namespace simulation

#endif // SIMULATION_TYPES_H_