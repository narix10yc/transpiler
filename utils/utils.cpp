#include "utils/utils.h"

bool utils::isPermutation(const std::vector<int>& v) {
    auto copy = v;
    std::sort(copy.begin(), copy.end());
    for (size_t i = 0, S = copy.size(); i < S; i++) {
        if (copy[i] != i)
            return false;
    }
    return true;
}

uint64_t utils::pdep64(uint64_t src, uint64_t mask) {
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

uint64_t utils::pdep64(uint64_t src, uint64_t mask, int nbits) {
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

void utils::timedExecute(std::function<void()> f, const char* msg) {
    using clock = std::chrono::high_resolution_clock;
    auto tic = clock::now();
    f();
    auto tok = clock::now();
    const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count();
    std::cerr << "-- (" << t_ms << " ms) " << msg << "\n";
    return;
}

std::ostream& utils::print_complex(std::ostream& os, std::complex<double> c, int precision) {
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
