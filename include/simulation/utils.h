#ifndef SIMULATION_UTILS_H_
#define SIMULATION_UTILS_H_

#include "cmath"

namespace simulation::utils {

double approximate(double x, double thres=1e-8) {
    if (abs(x) < thres)
        return 0;
    if (abs(x - 1) < thres)
        return 1;
    if (abs(x + 1) < thres)
        return -1;
    return x;
}

/// @brief Kronecker product
// template<typename real_t>
// ComplexMatrix4<real_t>
// kron(ComplexMatrix2<real_t> l, ComplexMatrix2<real_t> r) {
//     const auto& Lr = &(l.real); const auto& Rr = &(r.real);
//     const auto& Li = &(l.imag); const auto& Ri = &(r.imag);
//     return {
//         {
//             Lr[0]*Rr[0], Lr[0]*Rr[1],  Lr[1]*Rr[0], Lr[1]*Rr[1],
//             Lr[0]*Rr[2], Lr[0]*Rr[3],  Lr[1]*Rr[2], Lr[1]*Rr[3],
//             Lr[2]*Rr[0], Lr[2]*Rr[1],  Lr[3]*Rr[0], Lr[3]*Rr[1],
//             Lr[2]*Rr[2], Lr[2]*Rr[3],  Lr[3]*Rr[2], Lr[3]*Rr[3],
//         },
//         {
//             Li[0]*Ri[0], Li[0]*Ri[1],  Li[1]*Ri[0], Li[1]*Ri[1],
//             Li[0]*Ri[2], Li[0]*Ri[3],  Li[1]*Ri[2], Li[1]*Ri[3],
//             Li[2]*Ri[0], Li[2]*Ri[1],  Li[3]*Ri[0], Li[3]*Ri[1],
//             Li[2]*Ri[2], Li[2]*Ri[3],  Li[3]*Ri[2], Li[3]*Ri[3],
//         }
//     };
// }

} // namespace simulation

#endif // SIMULATION_UTILS_H_
