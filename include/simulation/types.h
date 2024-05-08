#ifndef SIMULATION_TYPES_H_
#define SIMULATION_TYPES_H_

#include <optional>
#include <cmath>
#include <initializer_list>
#include <array>
#include <iostream>
#include <iomanip>

namespace simulation {

namespace ir {
    enum class RealTy {
    Double, Float
};

enum class AmpFormat {
    Separate, Alternating
};

/// @brief Sperial complex matrix representation used in IR generation.
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

/// @brief Sperial complex matrix representation used in IR generation.
/// Its entries are ints, only with special meanings when equal to +1, -1, or 0.
/// All other cases are treated equal.
class ComplexMatrix4 {
public:
    std::array<int, 16> real, imag;
    ComplexMatrix4() = delete;
    explicit ComplexMatrix4(std::array<int, 16> real, std::array<int, 16> imag)
        : real(real), imag(imag) {}
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

    
}; // OptionalComplexMatrix2

template<typename real_ty=double>
class ComplexMatrix2 {
public:
    std::array<real_ty, 4> real, imag;

    ComplexMatrix2() : real(), imag() {}
    
    ComplexMatrix2(std::array<real_ty, 4> real, std::array<real_ty, 4> imag)
        : real(real), imag(imag) {}

    static ComplexMatrix2 GetIdentity() {
        return {{1,0,0,1}, {}};
    }

    ir::ComplexMatrix2 ToIRMatrix(real_ty threshold=1e-8) const {
        std::array<int, 4> realArr, imagArr;
        for (size_t i = 0; i < 4; i++) {
            if (std::abs(real[i] - 1) < threshold)
                realArr[i] = 1;
            else if (std::abs(real[i] + 1) < threshold)
                realArr[i] = -1;
            else if (std::abs(real[i]) < threshold)
                realArr[i] = 0;
            else
                realArr[i] = 2;

            if (std::abs(imag[i] - 1) < threshold)
                imagArr[i] = 1;
            else if (std::abs(imag[i] + 1) < threshold)
                imagArr[i] = -1;
            else if (std::abs(imag[i]) < threshold)
                imagArr[i] = 0;
            else
                imagArr[i] = 2;    
        }
        return ir::ComplexMatrix2 { realArr, imagArr };
    }

    static ComplexMatrix2
    FromEulerAngles(std::optional<double> theta, std::optional<double> lambd,
                    std::optional<double> phi, double thres=1e-8);
};

template<typename real_ty=double>
class ComplexMatrix4 {
public:
    std::array<real_ty, 16> real, imag;

    ComplexMatrix4() : real(), imag() {}

    ComplexMatrix4(std::array<real_ty, 16> real, std::array<real_ty, 16> imag)
        : real(real), imag(imag) {}

    static ComplexMatrix4 GetIdentity() {
        return {{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, {}};
    }

    ir::ComplexMatrix4 toIRMatrix(real_ty threshold=1e-8) const {
        std::array<int, 16> realArr, imagArr;
        for (size_t i = 0; i < 16; i++) {
            if (std::abs(real[i] - 1) < threshold)
                realArr[i] = 1;
            else if (std::abs(real[i] + 1) < threshold)
                realArr[i] = -1;
            else if (std::abs(real[i]) < threshold)
                realArr[i] = 0;
            else
                realArr[i] = 2;

            if (std::abs(imag[i] - 1) < threshold)
                imagArr[i] = 1;
            else if (std::abs(imag[i] + 1) < threshold)
                imagArr[i] = -1;
            else if (std::abs(imag[i]) < threshold)
                imagArr[i] = 0;
            else
                imagArr[i] = 2;
        }
        return ir::ComplexMatrix4 { realArr, imagArr };
    }

    ComplexMatrix4 matmul(ComplexMatrix4 other) {
        ComplexMatrix4 newMat {};
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
        
};


class U3Gate {
public:
    ComplexMatrix2<> mat;
    uint8_t k;
public:
    U3Gate(ComplexMatrix2<> mat, uint8_t k) : mat(mat), k(k) {}
};

class U2qGate {
public:
    ComplexMatrix4<> mat;
    uint8_t q0, q1;
public:
    U2qGate(ComplexMatrix4<> mat, uint8_t q0, uint8_t q1)
        : mat(mat), q0(q0), q1(q1) {}
};






} // namespace simulation

#endif // SIMULATION_TYPES_H_