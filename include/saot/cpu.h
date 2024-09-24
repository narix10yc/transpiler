#ifndef SAOT_CPU_H
#define SAOT_CPU_H

#include "saot/CircuitGraph.h"
#include "simulation/ir_generator.h"

#include <iomanip>

namespace saot::cpu {

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

} // namespace saot::cpu

#endif // SAOT_CPU_H