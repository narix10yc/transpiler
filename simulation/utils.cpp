#include "simulation/types.h"
#include <random>

using namespace simulation;

double randomAngle() {
    std::random_device rd;
    std::mt19937 gen { rd() };
    std::uniform_real_distribution<double> d { -M_PI, M_PI };
    return d(gen);
}

ComplexMatrix2<double> randomComplexMatrix2() {
    std::random_device rd;
    std::mt19937 gen { rd() };
    std::normal_distribution<double> d {0, 1.0}; // Normal(0, 1)

    std::array<double, 4> real;
    std::array<double, 4> imag;

    for (size_t i = 0; i < 4; i++) {
        real[i] = d(gen);
        imag[i] = d(gen);
    }

    return {real, imag};
}

ComplexMatrix4<double> randomComplexMatrix4() {
    std::random_device rd;
    std::mt19937 gen { rd() };
    std::normal_distribution<double> d {0, 1.0}; // Normal(0, 1)

    std::array<double, 16> real;
    std::array<double, 16> imag;

    for (size_t i = 0; i < 16; i++) {
        real[i] = d(gen);
        imag[i] = d(gen);
    }

    return {real, imag};
}