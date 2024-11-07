#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cassert>

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
static std::ostream& printVector(
        const std::vector<T>& v, std::ostream& os = std::cerr) {
    if (v.empty())
        return os << "[]";
    auto it = v.cbegin();
    os << "[" << *it;
    while (++it != v.cend())
        os << "," << *it; 
    return os << "]";
}

// The printer is expected to take inputs (const T&, std::ostream&)
template<typename T, typename Printer_T>
static std::ostream& printVectorWithPrinter(
        const std::vector<T>& v, Printer_T f,
        std::ostream& os = std::cerr) {
    if (v.empty())
        return os << "[]";
    auto it = v.cbegin();
    f(*it, os << "[");
    while (++it != v.cend())
        f(*it, os << ",");
    return os << "]";
}

// @return true if elem is in vec
template<typename T>
static void pushBackIfNotInVector(std::vector<T>& vec, T elem) {
    for (const auto& e : vec) {
        if (e == elem)
            return;
    }
    vec.push_back(elem);
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

static uint64_t pdep64(uint64_t src, uint64_t mask) {
    unsigned k = 0;
    uint64_t dst = 0ULL;
    for (unsigned i = 0; i < 64; ++i) {
        if (mask & (1ULL << i)) {
            if (src & (1ULL << i))
                dst |= (1 << k);
            ++k;
        }
    }
    return dst;
}

static uint64_t pdep64(uint64_t src, uint64_t mask, int nbits) {
    unsigned k = 0;
    uint64_t dst = 0ULL;
    for (unsigned i = 0; i < nbits; ++i) {
        if (mask & (1ULL << i)) {
            if (src & (1ULL << i))
                dst |= (1 << k);
            ++k;
        }
    }
    return dst;
}

class as0b {
    uint64_t v;
    int nbits;
public:
    as0b(uint64_t v, int nbits) : v(v), nbits(nbits) {
        assert(nbits > 0 && nbits <= 64);
    }

    friend std::ostream& operator<<(std::ostream& os, const as0b& n) {
        for (int i = n.nbits-1; i >= 0; --i)
            os.put((n.v & (1 << i)) ? '1' : '0');
        return os;
    }
};

} // namespace utils

#endif // UTILS_UTILS_H