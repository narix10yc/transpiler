#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <vector>
#include <iomanip>

namespace utils {

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

} // namespace utils

#endif // UTILS_UTILS_H