#ifndef SIMULATION_TPLT_H
#define SIMULATION_TPLT_H

#include <cstdlib>
#include "simulation/types.h"

namespace simulation::tplt {

template<typename real_ty>
void applySingleQubit(real_ty* real,
                      real_ty* imag,
                      const ComplexMatrix2<real_ty>& mat,
                      size_t nqubits,
                      size_t k) {
    size_t K = 1 << k;
    size_t N = 1 << nqubits;
    real_ty x_real, x_imag, y_real, y_imag;

    for (size_t t = 0; t < N; t += (2*K)) {
    for (size_t tt = 0; tt < K; tt++) {
        x_real = mat.real[0] * real[t+tt] + mat.real[1] * real[t+tt+K]
                -mat.imag[0] * imag[t+tt] - mat.imag[1] * imag[t+tt+K];
        x_imag = mat.real[0] * imag[t+tt] + mat.real[1] * imag[t+tt+K]
                +mat.imag[0] * real[t+tt] + mat.imag[1] * real[t+tt+K];
        y_real = mat.real[2] * real[t+tt] + mat.real[3] * real[t+tt+K]
                -mat.imag[2] * imag[t+tt] - mat.imag[3] * imag[t+tt+K];
        y_imag = mat.real[2] * imag[t+tt] + mat.real[3] * imag[t+tt+K]
                +mat.imag[2] * real[t+tt] + mat.imag[3] * real[t+tt+K];
        real[t+tt] = x_real;
        imag[t+tt] = x_imag;
        real[t+tt+K] = y_real;
        imag[t+tt+K] = y_imag;
    } }
}

template<typename real_ty>
void applySingleQubitQuEST(real_ty* real,
                           real_ty* imag,
                           const ComplexMatrix2<real_ty>& mat,
                           size_t nqubits,
                           size_t k) {
    size_t K = 1 << k;
    size_t sizeBlock = 2 * K;
    size_t N = 1 << nqubits;
    real_ty x_real, x_imag, y_real, y_imag;
    size_t thisBlock, alpha, beta;

    for (size_t t = 0; t < (N>>1); t++) {
        thisBlock = t / K;
        alpha = thisBlock * sizeBlock + t % K;
        beta = alpha + K;

        real[alpha] = mat.real[0] * real[alpha] + mat.real[1] * real[beta]
                     -mat.imag[0] * imag[alpha] - mat.imag[1] * imag[beta];
        imag[alpha] = mat.real[0] * imag[alpha] + mat.real[1] * imag[beta]
                     +mat.imag[0] * real[alpha] + mat.imag[1] * real[beta];
        real[beta] = mat.real[2] * real[alpha] + mat.real[3] * real[beta]
                    -mat.imag[2] * imag[alpha] - mat.imag[3] * imag[beta];
        imag[beta] = mat.real[2] * imag[alpha] + mat.real[3] * imag[beta]
                    +mat.imag[2] * real[alpha] + mat.imag[3] * real[beta];
    }
}

template<typename real_ty, size_t k>
void applySingleQubitTemplate(real_ty* real,
                           real_ty* imag,
                           const ComplexMatrix2<real_ty>& mat,
                           size_t nqubits) {
    size_t K = 1 << k;
    size_t N = 1 << nqubits;
    real_ty x_real, x_imag, y_real, y_imag;

    for (size_t t = 0; t < N; t += (2*K)) {
    for (size_t tt = 0; tt < K; tt++) {
        x_real = mat.real[0] * real[t+tt] + mat.real[1] * real[t+tt+K]
                -mat.imag[0] * imag[t+tt] - mat.imag[1] * imag[t+tt+K];
        x_imag = mat.real[0] * imag[t+tt] + mat.real[1] * imag[t+tt+K]
                +mat.imag[0] * real[t+tt] + mat.imag[1] * real[t+tt+K];
        y_real = mat.real[2] * real[t+tt] + mat.real[3] * real[t+tt+K]
                -mat.imag[2] * imag[t+tt] - mat.imag[3] * imag[t+tt+K];
        y_imag = mat.real[2] * imag[t+tt] + mat.real[3] * imag[t+tt+K]
                +mat.imag[2] * real[t+tt] + mat.imag[3] * real[t+tt+K];
        real[t+tt] = x_real;
        imag[t+tt] = x_imag;
        real[t+tt+K] = y_real;
        imag[t+tt+K] = y_imag;
    } }
}

template<typename real_ty>
void applyTwoQubit(real_ty* real,
                   real_ty* imag,
                   const ComplexMatrix4<real_ty>& mat,
                   size_t nqubits,
                   size_t k, size_t l) {
    size_t K = 1 << k;
    size_t L = 1 << l;
    size_t N = 1 << nqubits;
    real_ty *amp_pt_r, *amp_pt_i;
    real_ty amp_r[4], amp_i[4];

    for (size_t t = 0; t < N; t += (K << 1)) {
    for (size_t tt = 0; tt < K; tt += (L << 1)) {
    for (size_t ttt = 0; ttt < L; ttt++) {
        amp_pt_r = real + t + tt + ttt;
        amp_pt_i = imag + t + tt + ttt;
        amp_r[0] = (mat.real[0] * amp_pt_r[0]   - mat.imag[0] * amp_pt_i[0]) + 
                   (mat.real[1] * amp_pt_r[L]   - mat.imag[1] * amp_pt_i[L]) +
                   (mat.real[2] * amp_pt_r[K]   - mat.imag[2] * amp_pt_i[K]) +
                   (mat.real[3] * amp_pt_r[L|K] - mat.imag[3] * amp_pt_i[L|K]);

        amp_r[1] = (mat.real[4] * amp_pt_r[0]   - mat.imag[4] * amp_pt_i[0]) + 
                   (mat.real[5] * amp_pt_r[L]   - mat.imag[5] * amp_pt_i[L]) +
                   (mat.real[6] * amp_pt_r[K]   - mat.imag[6] * amp_pt_i[K]) +
                   (mat.real[7] * amp_pt_r[L|K] - mat.imag[7] * amp_pt_i[L|K]);

        amp_r[2] = (mat.real[8] * amp_pt_r[0]   - mat.imag[8] * amp_pt_i[0]) + 
                   (mat.real[9] * amp_pt_r[L]   - mat.imag[9] * amp_pt_i[L]) +
                   (mat.real[10] * amp_pt_r[K]   - mat.imag[10] * amp_pt_i[K]) +
                   (mat.real[11] * amp_pt_r[L|K] - mat.imag[11] * amp_pt_i[L|K]);

        amp_r[3] = (mat.real[12] * amp_pt_r[0]   - mat.imag[12] * amp_pt_i[0]) + 
                   (mat.real[13] * amp_pt_r[L]   - mat.imag[13] * amp_pt_i[L]) +
                   (mat.real[14] * amp_pt_r[K]   - mat.imag[14] * amp_pt_i[K]) +
                   (mat.real[15] * amp_pt_r[L|K] - mat.imag[15] * amp_pt_i[L|K]);

        amp_i[0] = (mat.real[0] * amp_pt_i[0]   + mat.imag[0] * amp_pt_r[0]) + 
                   (mat.real[1] * amp_pt_i[L]   + mat.imag[1] * amp_pt_r[L]) +
                   (mat.real[2] * amp_pt_i[K]   + mat.imag[2] * amp_pt_r[K]) +
                   (mat.real[3] * amp_pt_i[L|K] + mat.imag[3] * amp_pt_r[L|K]);

        amp_i[1] = (mat.real[4] * amp_pt_i[0]   + mat.imag[4] * amp_pt_r[0]) + 
                   (mat.real[5] * amp_pt_i[L]   + mat.imag[5] * amp_pt_r[L]) +
                   (mat.real[6] * amp_pt_i[K]   + mat.imag[6] * amp_pt_r[K]) +
                   (mat.real[7] * amp_pt_i[L|K] + mat.imag[7] * amp_pt_r[L|K]);

        amp_i[2] = (mat.real[8] * amp_pt_i[0]   + mat.imag[8] * amp_pt_r[0]) + 
                   (mat.real[9] * amp_pt_i[L]   + mat.imag[9] * amp_pt_r[L]) +
                   (mat.real[10] * amp_pt_i[K]   + mat.imag[10] * amp_pt_r[K]) +
                   (mat.real[11] * amp_pt_i[L|K] + mat.imag[11] * amp_pt_r[L|K]);

        amp_i[3] = (mat.real[12] * amp_pt_i[0]   + mat.imag[12] * amp_pt_r[0]) + 
                   (mat.real[13] * amp_pt_i[L]   + mat.imag[13] * amp_pt_r[L]) +
                   (mat.real[14] * amp_pt_i[K]   + mat.imag[14] * amp_pt_r[K]) +
                   (mat.real[15] * amp_pt_i[L|K] + mat.imag[15] * amp_pt_r[L|K]);

        amp_pt_r[0] = amp_r[0];
        amp_pt_r[L] = amp_r[1];
        amp_pt_r[K] = amp_r[2];
        amp_pt_r[L|K] = amp_r[3];
        amp_pt_i[0] = amp_i[0];
        amp_pt_i[L] = amp_i[1];
        amp_pt_i[K] = amp_i[2];
        amp_pt_i[L|K] = amp_i[3];
    } } }
}


} // namespace simultion::tplt


#endif // SIMULATION_TPLT_H