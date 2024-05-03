#include "simulation/types.h"

#include <functional>

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