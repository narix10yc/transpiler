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


inline uint64_t flipBit(uint64_t number, int bitInd) {
    return (number ^ (1LL << bitInd));
}

inline uint64_t insertZeroBit(uint64_t number, int index) {
    uint64_t left, right;
    left = (number >> index) << index;
    right = number - left;
    return (left << 1) ^ right;
}

inline uint64_t insertTwoZeroBits(uint64_t number, int bit1, int bit2) {
    int small = (bit1 < bit2) ? bit1 : bit2;
    int big = (bit1 < bit2) ? bit2 : bit1;
    return insertZeroBit(insertZeroBit(number, small), big);
}

/// @brief Apply a general two-qubit gate.
/// @param l less significant qubit
/// @param k more significant qubit
template<typename real_ty>
void applyTwoQubitQuEST(real_ty* real,
                   real_ty* imag,
                   const ComplexMatrix4<real_ty>& mat,
                   size_t nqubits,
                   size_t k, size_t l) {
    size_t nTasks = 1 << (nqubits - 2);
    size_t idx00, idx01, idx10, idx11;

    real_ty re00, re01, re10, re11, im00, im01, im10, im11;

    for (size_t t = 0; t < nTasks; t++) {
        idx00 = insertTwoZeroBits(t, k, l);
        idx01 = flipBit(idx00, l);
        idx10 = flipBit(idx00, k);
        idx11 = flipBit(idx01, k);

        re00 = real[idx00]; re01 = real[idx01];
        re10 = real[idx10]; re11 = real[idx11];
        im00 = imag[idx00]; im01 = imag[idx01];
        im10 = imag[idx10]; im11 = imag[idx11];

        real[idx00] = (mat.real[0] * re00 - mat.imag[0] * im00) + 
                      (mat.real[1] * re01 - mat.imag[1] * im01) +
                      (mat.real[2] * re10 - mat.imag[2] * im10) +
                      (mat.real[3] * re11 - mat.imag[3] * im11);

        real[idx01] = (mat.real[4] * re00 - mat.imag[4] * im00) + 
                      (mat.real[5] * re01 - mat.imag[5] * im01) +
                      (mat.real[6] * re10 - mat.imag[6] * im10) +
                      (mat.real[7] * re11 - mat.imag[7] * im11);

        real[idx10] = (mat.real[8] * re00 - mat.imag[8] * im00) + 
                      (mat.real[9] * re01 - mat.imag[9] * im01) +
                      (mat.real[10] * re10 - mat.imag[10] * im10) +
                      (mat.real[11] * re11 - mat.imag[11] * im11);

        real[idx11] = (mat.real[12] * re00 - mat.imag[12] * im00) + 
                      (mat.real[13] * re01 - mat.imag[13] * im01) +
                      (mat.real[14] * re10 - mat.imag[14] * im10) +
                      (mat.real[15] * re11 - mat.imag[15] * im11);

        imag[idx00] = (mat.real[0] * im00 + mat.imag[0] * re00) + 
                      (mat.real[1] * im01 + mat.imag[1] * re01) +
                      (mat.real[2] * im10 + mat.imag[2] * re10) +
                      (mat.real[3] * im11 + mat.imag[3] * re11);

        imag[idx01] = (mat.real[4] * im00 + mat.imag[4] * re00) + 
                      (mat.real[5] * im01 + mat.imag[5] * re01) +
                      (mat.real[6] * im10 + mat.imag[6] * re10) +
                      (mat.real[7] * im11 + mat.imag[7] * re11);

        imag[idx10] = (mat.real[8] * im00 + mat.imag[8] * re00) + 
                      (mat.real[9] * im01 + mat.imag[9] * re01) +
                      (mat.real[10] * im10 + mat.imag[10] * re10) +
                      (mat.real[11] * im11 + mat.imag[11] * re11);

        imag[idx11] = (mat.real[12] * im00 + mat.imag[12] * re00) + 
                      (mat.real[13] * im01 + mat.imag[13] * re01) +
                      (mat.real[14] * im10 + mat.imag[14] * re10) +
                      (mat.real[15] * im11 + mat.imag[15] * re11);
    }
}


} // namespace simultion::tplt


#endif // SIMULATION_TPLT_H