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
};

class CodeGeneratorCPU {
private:
    std::string fileName;
    CodeGeneratorCPUConfig config;
public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : fileName(fileName), 
          config({.s=1, .precision=64, .multiThreaded=false,
                  .installTimer=false, .overrideNqubits=-1}) {}

    void generate(const circuit_graph::CircuitGraph& graph, int verbose=0);

    void config_s(int s) { config.s = s; }
    void config_precision(int p) { config.precision = p; }
    void config_multiThreaded(bool on) { config.multiThreaded = on; }
    void config_timer(bool install) { config.installTimer = install; }
    void config_nqubits(int n) { config.overrideNqubits = n; }

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