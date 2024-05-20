#include "simulation/types.h"

#include <functional>
#include <sstream>

using namespace simulation;

namespace {
uint32_t f(int x) {
    switch (x) {
        case 0: return 0;
        case 1: return 1;
        case -1: return 2;
        default: return 3;
    }
}

int f_inv(uint32_t y) {
    switch (y & 3) {
        case 0: return 0;
        case 1: return 1;
        case 2: return -1;
        default: return 2;
    }
}

int approximate(std::optional<double> a, double thres=1e-8) {
    if (!a.has_value()) return 2;
    double v = a.value();
    if (abs(v) < thres) return 0;
    if (abs(v - 1) < thres) return 1;
    if (abs(v + 1) < thres) return -1;
    return 2;
}
} // anonymous namespace


OptionalComplexMatrix2
OptionalComplexMatrix2::FromEulerAngles(std::optional<double> theta,
                                        std::optional<double> phi, 
                                        std::optional<double> lambd) {
    std::optional<double> ar, br, cr, dr, bi, ci, di;
    std::optional<double> ctheta, stheta, clambd, slambd, cphi, sphi,
                            cphi_lambd, sphi_lambd;
    if (theta.has_value()) {
        ctheta = cos(theta.value() * 0.5);
        stheta = sin(theta.value() * 0.5);
    }
    if (lambd.has_value()) {
        clambd = cos(lambd.value());
        slambd = sin(lambd.value());
    }
    if (phi.has_value()) {
        sphi = sin(phi.value());
        cphi = cos(phi.value());
    }
    if (phi.has_value() && lambd.has_value()) {
        cphi_lambd = cos(phi.value() + lambd.value());
        sphi_lambd = sin(phi.value() + lambd.value());
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

ir::ComplexMatrix2
OptionalComplexMatrix2::ToIRMatrix(double thres) const {
    return ir::ComplexMatrix2 {
        { approximate(ar, thres), approximate(br, thres),
          approximate(cr, thres), approximate(dr, thres) }, 
        { approximate(ai, thres), approximate(bi, thres),
          approximate(ci, thres), approximate(di, thres) }};
}

ir::U3Gate ir::U3Gate::FromID(uint32_t id) {
    uint8_t qubit = static_cast<uint8_t>((id >> 24) & 15);
    ir::ComplexMatrix2
      mat {{ f_inv(id >> 14), f_inv(id >> 12), f_inv(id >> 10), f_inv(id >> 8) },
           { f_inv(id >> 6),  f_inv(id >> 4),  f_inv(id >> 2),  f_inv(id >> 0) }};
    
    return ir::U3Gate { qubit, mat };
}

uint32_t ir::U3Gate::getID() const {
    uint32_t id = 0;
    id += f(mat.imag[3]);
    id += f(mat.imag[2]) << 2;
    id += f(mat.imag[1]) << 4;
    id += f(mat.imag[0]) << 6;
    id += f(mat.real[3]) << 8;
    id += f(mat.real[2]) << 10;
    id += f(mat.real[1]) << 12;
    id += f(mat.real[0]) << 14;
    id += static_cast<uint32_t>(k) << 24;
    return id;
}

std::string ir::U3Gate::getRepr() const {
    std::stringstream ss;
    ss << "u3_k" << static_cast<int>(k) << "_"
       << f(mat.real[0]) << f(mat.real[1]) << f(mat.real[2]) << f(mat.real[3])
       << f(mat.imag[0]) << f(mat.imag[1]) << f(mat.imag[2]) << f(mat.imag[3]);
    return ss.str();
}

ir::ComplexMatrix2 
ir::ComplexMatrix2::FromEulerAngles(std::optional<double> theta,
                                    std::optional<double> phi,
                                    std::optional<double> lambd,
                                    double thres) {
    return OptionalComplexMatrix2::FromEulerAngles(theta, phi, lambd).ToIRMatrix(thres);
}

std::string ir::U2qGate::getRepr() const {
    std::stringstream ss;
    ss << "u2q_k" << static_cast<int>(qLarge) << "l" << static_cast<int>(qSmall)
       << "_" << std::hex << std::setfill('0') << std::setw(16) << mat;
    return ss.str();
}