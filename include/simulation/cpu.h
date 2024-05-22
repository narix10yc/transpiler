#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>

#include "simulation/ir_generator.h"
#include "qch/ast.h"

namespace simulation {

class CPUGenContext {
    simulation::IRGenerator irGenerator;
    std::string fileName;
public:
    struct kernel {
        std::string name;
        unsigned nqubits;
    };
    std::map<uint32_t, std::string> u3GateMap;
    std::map<std::string, std::string> u2qGateMap;
    std::vector<std::array<double, 8>> u3Params;
    std::vector<std::array<double, 32>> u2qParams;
    std::vector<kernel> kernels;

    CPUGenContext(unsigned vecSizeInBits, std::string fileName)
        : irGenerator(vecSizeInBits),
          fileName(fileName) {}
    
    void setRealTy(ir::RealTy ty) { irGenerator.setRealTy(ty); }
    void setAmpFormat(ir::AmpFormat format) { irGenerator.setAmpFormat(format); }

    simulation::IRGenerator& getGenerator() { return irGenerator; }

    ir::RealTy getRealTy() const { return irGenerator.realTy; }
    ir::AmpFormat getAmpFormat() const { return irGenerator.ampFormat; }

    void setF32() { irGenerator.setRealTy(ir::RealTy::Float); }
    void setF64() { irGenerator.setRealTy(ir::RealTy::Double); }
    
    void setAlternatingAmpFormat() {
        irGenerator.setAmpFormat(ir::AmpFormat::Alternating);
    }
    void setSeparateAmpFormat() {
        irGenerator.setAmpFormat(ir::AmpFormat::Separate);
    }

    void generate(qch::ast::RootNode& root) {
        root.genCPU(*this);

        // shell File
        auto shellFileName = fileName + ".sh";
        std::ofstream shellFile {shellFileName};
        if (!shellFile.is_open()) {
            std::cerr << "Failed to open " << shellFileName << "\n";
            return;
        }
        std::cerr << "shell script will be written to: " << shellFileName << "\n";
        shellFile.close();

        // header file
        auto hFileName = fileName + ".h";
        std::ofstream hFile {hFileName};
        if (!hFile.is_open()) {
            std::cerr << "Failed to open " << hFileName << "\n";
            return;
        }
        std::cerr << "header file will be written to: " << hFileName << "\n";

        std::string typeStr =
            (getRealTy() == ir::RealTy::Double) ? "double" : "float";

        hFile << "#include <cstdint>\n\n";

        // declaration
        hFile << "extern \"C\" {\n";
        for (auto k : kernels) {
            hFile << "void " << k.name << "("
                  << typeStr << "*, " << typeStr << "*, int64_t, int64_t, "
                  << "const " << typeStr << "*);\n";
        }
        hFile << "}\n\n";

        // u3 param
        hFile << std::setprecision(16);
        hFile << "static const " << typeStr << " _u3Params[] = {\n";
        for (auto arr : u3Params) {
            for (auto p : arr) {
                hFile << p << ",";
            }
            hFile << "\n";
        }
        hFile << "};\n\n";

        // u2q param
        hFile << "static const " << typeStr << " _u2qParams[] = {\n";
        for (auto arr : u2qParams) {
            for (auto p : arr) {
                hFile << p << ",";
            }
            hFile << "\n";
        }
        hFile << "};\n\n";

        // simulate_ciruit
        hFile << "void simulate_circuit("
              << typeStr << "* real, " << typeStr << "* imag, "
              << "unsigned nqubits) {\n";

        unsigned u3Idx = 0, u2qIdx = 0;
        for (auto k : kernels) {
            if (k.nqubits == 1) {
                hFile << k.name << "(real, imag, 0, "
                      << "1ULL<<(nqubits-" << irGenerator.vecSizeInBits << "-1)"
                      << ", _u3Params+" << 8*u3Idx << ");\n";
                u3Idx++; 
            } else {
                hFile << k.name << "(real, imag, 0, "
                      << "1ULL<<(nqubits-" << irGenerator.vecSizeInBits << "-2)"
                      << ", _u2qParams+" << 36*u2qIdx << ");\n";
                u2qIdx++; 
            }
        }
        hFile << "}\n\n";

        hFile.close();

        // IR file
        std::error_code EC;
        auto irName = fileName + ".ll";
        auto irFile = llvm::raw_fd_ostream(irName, EC);
        std::cerr << "IR file will be written to: " << irName << "\n";

        irGenerator.getModule().print(irFile, nullptr);


        irFile.close();
    }
};

} // namespace simulation

#endif // SIMULATION_CPU_H_