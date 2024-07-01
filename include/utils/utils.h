#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <vector>

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

} // namespace utils

#endif // UTILS_UTILS_H