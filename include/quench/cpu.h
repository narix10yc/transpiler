#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"

namespace quench::cpu {

class CodeGeneratorCPU {
private:
    struct config_t {
        std::string fileName;
        int s;
        bool installTimer;
        int overrideNqubits;
    } config;

public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : config({.fileName=fileName, .s=1,
                  .installTimer=false, .overrideNqubits=-1}) {}

    void generate(const circuit_graph::CircuitGraph& graph);

    void config_installTimer(bool b) {
        config.installTimer = b;
    }
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H