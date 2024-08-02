#ifndef QUENCH_CPU_H
#define QUENCH_CPU_H

#include "quench/CircuitGraph.h"
#include "utils/iocolor.h"

#include <iomanip>

namespace quench::cpu {

struct CodeGeneratorCPUConfig {
    int simd_s;
    double zeroSkipThreshold;
    double closeMatrixEntryThres;
    int precision;
    int verbose;
    bool multiThreaded;
    bool installTimer;
    int overrideNqubits;
    bool loadMatrixInEntry;
    bool loadVectorMatrix;
    bool usePDEP; // parallel bit deposite
    bool dumpIRToMultipleFiles;
    bool enablePrefetch;
    bool generateAltKernel;
    bool forceDenseKernel;

    std::ostream& display(std::ostream& os = std::cerr) const {
        os << Color::CYAN_FG << "== CodeGen Configuration ==\n" << Color::RESET
           << "SIMD s:      " << simd_s << "\n"
           << "Precision:   " << "f" << precision << "\n"
           << "Verbose:     " << verbose << "\n";
        
        os << "Multi-threading "
           << ((multiThreaded) ? "enabled" : "disabled")
           << ".\n";
        
        if (installTimer)
            os << "Timer installed\n";
        if (overrideNqubits > 0)
            os << "Override nqubits = " << overrideNqubits << "\n";
        
        os << "Detailed IR settings:\n"
           << "  zero skip threshold:      " << std::scientific << std::setprecision(4) << zeroSkipThreshold << "\n"
           << "  close matrix entry thres: ";
        if (closeMatrixEntryThres < 0)
            os << " disabled\n";
        else
            os << std::scientific << std::setprecision(4) << closeMatrixEntryThres << "\n";
        os << "  load matrix in entry:     " << ((loadMatrixInEntry) ? "true" : "false") << "\n"
           << "  load vector matrix:       " << ((loadVectorMatrix) ? "true" : "false") << "\n"
           << "  use PDEP:                 " << ((usePDEP) ? "true" : "false") << "\n"
           << "  dump IR to multi files:   " << ((dumpIRToMultipleFiles) ? "true" : "false") << "\n"
           << "  enable prefetch:          " << ((enablePrefetch) ? "true" : "false") << "\n"
           << "  generate alt kernel:      " << ((generateAltKernel) ? "true" : "false") << "\n"
           << "  force dense kernel:       " << ((forceDenseKernel) ? "true" : "false") << "\n";

        os << Color::CYAN_FG << "===========================\n" << Color::RESET;
        return os;
    }

};

class CodeGeneratorCPU {
private:
    std::string fileName;
public:
    CodeGeneratorCPU(const std::string& fileName = "gen_file")
        : fileName(fileName), 
          config({.simd_s=1,
                  .zeroSkipThreshold=1e-8,
                  .closeMatrixEntryThres=-1.0,
                  .precision=64,
                  .verbose=0,
                  .multiThreaded=false,
                  .installTimer=false,
                  .overrideNqubits=-1,
                  .loadMatrixInEntry=true,
                  .loadVectorMatrix=true,
                  .usePDEP=true,
                  .dumpIRToMultipleFiles=false,
                  .enablePrefetch=false,
                  .generateAltKernel=false,
                  .forceDenseKernel=false
                 }) {}

    CodeGeneratorCPUConfig config;

    /// @brief Generate IR
    /// @param forceInOrder: force generate IR according to block id 
    void generate(const circuit_graph::CircuitGraph& graph, bool forceInOrder=false);

    std::ostream& displayConfig(std::ostream& os = std::cerr) const {
        return config.display(os);
    }
    
};

} // namespace quench::cpu

#endif // QUENCH_CPU_H