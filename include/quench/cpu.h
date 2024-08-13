#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"
#include "simulation/ir_generator.h"

#include <iomanip>

namespace quench::cpu {

struct CodeGeneratorCPUConfig {
    bool multiThreaded;
    bool installTimer;
    // int overrideNqubits;
    // bool dumpIRToMultipleFiles;
    simulation::IRGeneratorConfig irConfig;

    std::ostream& display(int verbose = 1, std::ostream& os = std::cerr) const;
};

class CodeGeneratorCPU {
private:
    std::string fileName;
public:
    CodeGeneratorCPU(const CodeGeneratorCPUConfig& config, const std::string& fileName = "gen_file")
        : fileName(fileName), 
          config(config) {}

    CodeGeneratorCPUConfig config;

    /// @brief Generate IR
    /// @param forceInOrder: force generate IR according to block id 
    void generate(const circuit_graph::CircuitGraph& graph,
                  int debugLevel = 0, bool forceInOrder = false);

    std::ostream& displayConfig(int verbose = 1, std::ostream& os = std::cerr) const {
        return config.display(verbose, os);
    }
    
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H