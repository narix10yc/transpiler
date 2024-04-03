#include "simulation/irGen.h"
#include "openqasm/ast.h"
#include <map>

namespace {
uint32_t elemToID(std::optional<double> v) {
    if (!v.has_value())
        return 0b11;
    double value = v.value();
    if (value == 1)
        return 0b01;
    if (value == 0)
        return 0b00;
    if (value == -1)
        return 0b10;
    return 0b11;
}

std::optional<double> idToElem(uint32_t id) {
    switch (id & 3) {
        case 0: return 0;
        case 1: return 1;
        case 2: return -1;
        case 3: return {};
    }
    return {};
}
} // <anonymous> namespace

class U3Matrix {
public:
    std::optional<double> ar, br, cr, dr, bi, ci, di;
    uint8_t qubit;

    static U3Matrix FromID(uint32_t id) {
        uint8_t qubit = static_cast<uint8_t>(id & 0xF000);
        return { idToElem(id >> 12), idToElem(id >> 10),
                 idToElem(id >> 8), idToElem(id >> 6),
                 idToElem(id >> 4), idToElem(id >> 2),
                 idToElem(id >> 0), qubit };
    }

    /// @brief 32-bit id. From most to least significant: k (8-bit), 0 (10-bit),
    /// ar, br, cr, dr, bi, ci, di. Each number takes 2 bits following the rule: 
    /// +1 -> 01; 0 -> 00; -1 -> 10; others -> 11
    uint32_t getID() const {
        uint32_t id = 0;
        id += elemToID(di);
        id += elemToID(ci) << 2;
        id += elemToID(bi) << 4;
        id += elemToID(dr) << 6;
        id += elemToID(cr) << 8;
        id += elemToID(br) << 10;
        id += elemToID(ar) << 12;
        id += static_cast<uint32_t>(qubit) << 24;
        return id;
    }
};



class IRContext {
    std::map<uint32_t, std::string> gateMap;
    // IRGenerator

};

void generateFiles(openqasm::ast::RootNode &root) {

}

int main() {
    return 0;
}