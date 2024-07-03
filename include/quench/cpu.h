#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"

namespace quench::cpu {

class CodeGeneratorCPU {
private:
    struct config_t {
        std::string fileName;
        int s;
        int overrideNqubits;
    } config;

public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : config({fileName, .s=1, -1}) {}

    void generate(const circuit_graph::CircuitGraph& graph);
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H