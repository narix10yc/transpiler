#ifndef SIMULATION_TYPES_H_
#define SIMULATION_TYPES_H_

#include <optional>
#include <cmath>
#include <initializer_list>
#include <array>
#include <iostream>
#include <iomanip>

namespace simulation {

// static std::optional<double> optionalCos(std::optional<double> a) {
//     if (!a.has_value()) return {};
//     return { cos(a.value()) };
// }

// static std::optional<double> optionalSin(std::optional<double> a) {
//     if (!a.has_value()) return {};
//     return { sin(a.value()) };
// }

static double approxCos(double a, double thres=1e-8) {
    double v = cos(a);
    if (abs(v) < thres) { return 0; }
    if (abs(v - 1) < thres) { return 1; }
    if (abs(v + 1) < thres) { return -1; }
    return v; 
}

static double approxSin(double a, double thres=1e-8) {
    double v = sin(a);
    if (abs(v) < thres) { return 0; }
    if (abs(v - 1) < thres) { return 1; }
    if (abs(v + 1) < thres) { return -1; }
    return v; 
}

class OptionalComplexMat2x2 {
public:
    std::optional<double> ar, br, cr, dr, ai, bi, ci, di;
public:
    OptionalComplexMat2x2(std::optional<double> ar, 
    std::optional<double> br, std::optional<double> cr,
    std::optional<double> dr, std::optional<double> bi,
    std::optional<double> ci, std::optional<double> di) 
    : ar(ar), br(br), cr(cr), dr(dr), bi(bi), ci(ci), di(di) {}

    static OptionalComplexMat2x2 
    FromAngles(std::optional<double> theta, std::optional<double> phi, 
               std::optional<double> lambd, double thres=1e-8) {
        std::optional<double> ar, br, cr, dr, bi, ci, di;
        std::optional<double> ctheta, stheta, clambd, slambd, cphi, sphi,
                              cphi_lambd, sphi_lambd;
        if (theta.has_value()) {
            ctheta = approxCos(theta.value() * 0.5, thres);
            stheta = approxSin(theta.value() * 0.5, thres);
        }
        if (lambd.has_value()) {
            clambd = approxCos(lambd.value(), thres);
            slambd = approxSin(lambd.value(), thres);
        }
        if (phi.has_value()) {
            cphi = approxCos(phi.value(), thres);
            sphi = approxSin(phi.value(), thres);
        }
        if (phi.has_value() && lambd.has_value()) {
            cphi_lambd = approxCos(phi.value() + lambd.value(), thres);
            sphi_lambd = approxSin(phi.value() + lambd.value(), thres);
        }

        // ar: cos(theta/2)
        if (ctheta.has_value())
            ar.emplace(ctheta.value());

        // br: -cos(lambd) * sin(theta/2)
        if (clambd.has_value() && stheta.has_value()) {
            br.emplace(-clambd.value() * stheta.value());
        }
        if ((clambd.has_value() && clambd == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            br.emplace(0);
        }

        // cr: cos(phi) * sin(theta/2)
        if (cphi.has_value() && stheta.has_value()) {
            cr.emplace(cphi.value() * stheta.value());
        }
        if ((cphi.has_value() && cphi == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            cr.emplace(0);
        }

        // dr: cos(phi+lambd) * cos(theta/2)
        if (cphi_lambd.has_value() && ctheta.has_value()) {
            dr.emplace(cphi_lambd.value() * ctheta.value());
        }
        if ((cphi_lambd.has_value() && cphi_lambd == 0) || 
                 (ctheta.has_value() && ctheta == 0)) {
            dr.emplace(0);
        }

        // bi: -sin(lambd) * sin(theta/2)
        if (slambd.has_value() && stheta.has_value()) {
            bi.emplace(-slambd.value() * stheta.value());
        }
        if ((slambd.has_value() && slambd == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            bi.emplace(0);
        }

        // ci: sin(phi) * sin(theta/2)
        if (sphi.has_value() && stheta.has_value()) {
            ci.emplace(sphi.value() * stheta.value());
        }
        if ((sphi.has_value() && sphi == 0) || 
                 (stheta.has_value() && stheta == 0)) {
            ci.emplace(0);
        }

        // di: sin(phi+lambd) * cos(theta/2)
        if (sphi_lambd.has_value() && ctheta.has_value()) {
            di.emplace(sphi_lambd.value() * ctheta.value());
        }
        if ((sphi_lambd.has_value() && sphi_lambd == 0) || 
                 (ctheta.has_value() && ctheta == 0)) {
            di.emplace(0);
        }
        return { ar, br, cr, dr, bi, ci, di };
    }
}; // OptionalComplexMat2x2



class U3Gate {
private:
    static uint32_t elemToID(std::optional<double> v) {
        if (!v.has_value())
            return 0b11;
        double value = v.value();
        if (value == 1)
            return 0b01;
        if (value == 0)
            return 0b00;
        if (value == -1)
            return 0b10;
        return 0b11;
    }

    static std::optional<double> idToElem(uint32_t id) {
        switch (id & 3) {
            case 0: return 0;
            case 1: return 1;
            case 2: return -1;
            case 3: return {};
        }
        return {};
    }
public:
    OptionalComplexMat2x2 mat;
    uint8_t qubit;

    U3Gate(OptionalComplexMat2x2 mat, uint8_t qubit)
        : mat(mat), qubit(qubit) {}

    static U3Gate FromID(uint32_t id) {
        uint8_t qubit = static_cast<uint8_t>(id & 0xF000);
        OptionalComplexMat2x2 m { idToElem(id >> 12), 
        idToElem(id >> 10), idToElem(id >> 8), idToElem(id >> 6),
        idToElem(id >> 4), idToElem(id >> 2), idToElem(id >> 0) };

        return { m, qubit };
    }

    static U3Gate FromAngles(uint8_t qubit,
                             std::optional<double> theta,
                             std::optional<double> phi, 
                             std::optional<double> lambd,
                             double thres=1e-8) {
        auto mat = OptionalComplexMat2x2::FromAngles(theta, phi, lambd, thres);
        return { mat, qubit };
    }

    /// @brief 32-bit id. From most to least significant: k (8-bit), 0 (10-bit),
    /// ar, br, cr, dr, bi, ci, di. Each number takes 2 bits following the rule: 
    /// +1 -> 01; 0 -> 00; -1 -> 10; others -> 11
    uint32_t getID() const {
        uint32_t id = 0;
        id += elemToID(mat.di);
        id += elemToID(mat.ci) << 2;
        id += elemToID(mat.bi) << 4;
        id += elemToID(mat.dr) << 6;
        id += elemToID(mat.cr) << 8;
        id += elemToID(mat.br) << 10;
        id += elemToID(mat.ar) << 12;
        id += static_cast<uint32_t>(qubit) << 24;
        return id;
    }
}; // U3Gate


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


class U2qGate {
    ComplexMatrix4<> mat;
    int q0, q1;
public:
    U2qGate(ComplexMatrix4<> mat, int q0, int q1) : mat(mat), q0(q0), q1(q1) {}
};






} // namespace simulation

#endif // SIMULATION_TYPES_H_