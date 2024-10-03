#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace utils {

static bool isPermutation(const std::vector<int>& v) {
    auto copy = v;
    std::sort(copy.begin(), copy.end());
    for (size_t i = 0, S = copy.size(); i < S; i++) {
        if (copy[i] != i)
            return false;
    }
    return true;
}

template<typename T>
static bool isOrdered(const std::vector<T>& vec, bool ascending = true) {
    if (vec.empty())
        return true;
    
    if (ascending) {
        for (unsigned i = 0; i < vec.size() - 1; i++) {
            if (vec[i] > vec[i+1])
                return false;
        }
        return true;
    } else {
        for (unsigned i = 0; i < vec.size() - 1; i++) {
            if (vec[i] < vec[i+1])
                return false;
        }
        return true;
    }
}

static std::ostream&
print_complex(std::ostream& os, std::complex<double> c, int precision=3) {
    const double thres = 0.5 * std::pow(0.1, precision);
    if (c.real() >= -thres)
        os << " ";
    if (std::fabs(c.real()) < thres)
        os << "0." << std::string(precision, ' ');
    else
        os << std::fixed << std::setprecision(precision) << c.real();
    
    if (c.imag() >= -thres)
        os << "+";
    if (std::fabs(c.imag()) < thres)
        os << "0." << std::string(precision, ' ');
    else
        os << std::fixed << std::setprecision(precision) << c.imag();
    return os << "i";
}


template<typename T>
static std::ostream& printVector(const std::vector<T>& v, std::ostream& os = std::cerr) {
    if (v.empty())
        return os << "[]";
    os << "[";
    for (unsigned i = 0; i < v.size() - 1; i++)
        os << v[i] << ",";
    os << v.back() << "]";
    return os;
}

template<typename T = uint64_t>
static T insertZeroToBit(T x, int bit) {
    T maskLo = (1 << bit) - 1;
    T maskHi = ~maskLo;
    return (x & maskLo) + ((x & maskHi) << 1);
}

template<typename T = uint64_t>
static T insertOneToBit(T x, int bit) {
    T maskLo = (1 << bit) - 1;
    T maskHi = ~maskLo;
    return (x & maskLo) | ((x & maskHi) << 1) | (1 << bit);
}



} // namespace utils

#endif // UTILS_UTILS_H