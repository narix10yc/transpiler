#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

namespace quench::cpu {

struct CodeGeneratorCPUConfig {
    int s;
    int precision;
    bool multiThreaded;
    bool installTimer;
    int overrideNqubits;
    bool loadMatrixInEntry;
    bool loadVectorMatrix;
    bool usePDEP; // parallel bit deposite
};

class CodeGeneratorCPU {
private:
    std::string fileName;
public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : fileName(fileName), 
          config({.s=1, .precision=64, .multiThreaded=false,
                  .installTimer=false,
                  .overrideNqubits=-1,
                  .loadMatrixInEntry=true,
                  .loadVectorMatrix=true,
                  .usePDEP=true}) {}

    CodeGeneratorCPUConfig config;

    void generate(const circuit_graph::CircuitGraph& graph, int verbose=0);

    std::ostream& displayConfig(std::ostream& os = std::cerr) const {
        os << Color::CYAN_FG << "== CodeGen Configuration ==\n" << Color::RESET
           << "SIMD s:       " << config.s << "\n"
           << "Precision:    " << "f" << config.precision << "\n";
        
        os << "Multi-threading "
           << ((config.multiThreaded) ? "enabled" : "disabled")
           << ".\n";
        
        if (config.installTimer)
            os << "Timer installed\n";
        
        if (config.overrideNqubits > 0)
            os << "Override nqubits = " << config.overrideNqubits << "\n";
        
        os << Color::CYAN_FG << "===========================\n" << Color::RESET;
        return os;
    }
    
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H