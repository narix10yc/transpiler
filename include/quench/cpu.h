#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

namespace quench::cpu {

struct CodeGeneratorCPUConfig {
    int s;
    bool multiThreaded;
    int nthreads;
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
          config({.s=1, .multiThreaded=false, .nthreads=16,
                  .installTimer=false, .overrideNqubits=-1}) {}

    void generate(const circuit_graph::CircuitGraph& graph, int verbose=0);

    void config_s(int s) { config.s = s; }
    void config_multiThreaded(bool on) { config.multiThreaded = on; }
    void config_nthreads(int nthreads) { config.nthreads = nthreads; }
    void config_timer(bool install) { config.installTimer = install; }
    void config_nqubits(int n) { config.overrideNqubits = n; }

    std::ostream& displayConfig(std::ostream& os = std::cerr) const {
        os << Color::CYAN_FG << "== CodeGen Configuration ==\n" << Color::RESET
           << "SIMD s:       " << config.s << "\n";
        
        os << "Multi-threading ";
        if (config.multiThreaded)
            os << "enabled. nthreads = " << config.nthreads;
        else
            os << "disabled.";
        os << "\n";
        
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