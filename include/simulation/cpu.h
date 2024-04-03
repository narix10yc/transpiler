#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <map>
#include "simulation/irGen.h"
#include "openqasm/ast.h"


namespace simulation {

class CPUGenContext {
public:
    unsigned vecSizeInBits;
    std::map<uint32_t, std::string> gateMap;
    simulation::IRGenerator irGenerator;

    CPUGenContext(unsigned vecSizeInBits)
        : vecSizeInBits(vecSizeInBits), irGenerator(vecSizeInBits) {}

    void logError(std::string msg) {}

    void generate(const openqasm::ast::RootNode& root);
};

} // namespace simulation

#endif // SIMULATION_CPU_H_