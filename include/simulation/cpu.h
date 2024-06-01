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
    int nthreads;
    struct kernel {
        std::string kernelName;
        std::string irFuncName;
        unsigned nqubits;
    };
    std::vector<kernel> kernels;
    std::vector<std::array<double, 8>> u3Params;
    std::vector<std::array<double, 32>> u2qParams;
public:
    std::map<uint32_t, std::string> u3GateMap;
    std::map<std::string, std::string> u2qGateMap;

    CPUGenContext(unsigned vecSizeInBits, std::string fileName,
                  unsigned nthreads=1)
        : irGenerator(vecSizeInBits),
          fileName(fileName),
          nthreads(nthreads),
          kernels(),
          u3Params(), u2qParams(),
          u3GateMap(), u2qGateMap() {}
    
    void setRealTy(ir::RealTy ty) { irGenerator.setRealTy(ty); }
    void setAmpFormat(ir::AmpFormat format) { irGenerator.setAmpFormat(format); }
    void setNThreads(unsigned nthreads) { nthreads = nthreads; }

    void addU3Gate(const std::string& irFuncName,
                   const std::array<double, 8>& params) {
        u3Params.push_back(params);
        std::string kernelName = "kernel_" + std::to_string(kernels.size()) + "_u3";
        kernels.push_back({kernelName, irFuncName, 1});
    }

    void addU2qGate(const std::string& irFuncName,
                    const std::array<double, 32>& params) {
        u2qParams.push_back(params);
        std::string kernelName = "kernel_" + std::to_string(kernels.size()) + "_u2q";
        kernels.push_back({kernelName, irFuncName, 2});
    }

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

        if (nthreads > 1)
            hFile << "#include <thread>\n";
        hFile << "#include <cstdint>\n\n";

        // declaration
        hFile << "extern \"C\" {\n";
        for (const auto& k : kernels) {
            hFile << "void " << k.irFuncName << "("
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
        for (const auto& arr : u2qParams) {
            for (auto p : arr) {
                hFile << p << ",";
            }
            hFile << "\n";
        }
        hFile << "};\n\n";

        // inline wrapper (handle different nthreads)
        for (const auto& k : kernels) {
            if (k.nqubits == 1 || (k.nqubits == 2 && nthreads == 1)) {
                hFile << "inline void " << k.kernelName
                      << "(" << typeStr << "* real, " << typeStr << "* imag, "
                      << "uint64_t idxMax, const " << typeStr << "* m) {\n "
                      << k.irFuncName << "(real, imag, 0, idxMax, m);\n}\n";
            } else if (k.nqubits == 2 && nthreads > 1) {
                hFile << "inline void " << k.kernelName
                      << "(" << typeStr << "* real, " << typeStr << "* imag, "
                      << "const " << typeStr << "* m, uint64_t chunkSize, "
                      << "std::thread* threads){\n "
                      << "for (unsigned i = 0; i < " << nthreads << "; i++)\n  "
                      << "threads[i] = std::thread{" << k.irFuncName
                      << ", real, imag, i*chunkSize, (i+1)*chunkSize, m};\n "
                      << "for (unsigned i = 0; i < " << nthreads << "; i++)\n  "
                      << "threads[i].join();\n}\n";
            } else {
                assert(false && "?");
            }
        }
        hFile << "\n";

        // simulate_ciruit
        hFile << "void simulate_circuit("
              << typeStr << "* real, " << typeStr << "* imag, "
              << "unsigned nqubits) {\n";
        if (nthreads > 1)
            hFile << " std::thread threads[" << nthreads << "];\n";

        unsigned u3Idx = 0, u2qIdx = 0;
        for (const auto& k : kernels) {
            if (k.nqubits == 1) {
                hFile << " " << k.kernelName << "(real, imag, "
                      << "1ULL<<(nqubits-" << irGenerator.vecSizeInBits << "-1)"
                      << ", _u3Params+" << 8*u3Idx << ");\n";
                u3Idx++; 
            } else if (k.nqubits == 2 && nthreads == 1) {
                hFile << " " << k.kernelName << "(real, imag, "
                      << "1ULL<<(nqubits-" << irGenerator.vecSizeInBits << "-2)"
                      << ", _u2qParams+" << 32*u2qIdx << ");\n";
                u2qIdx++; 
            } else if (k.nqubits == 2 && nthreads > 1) {
                hFile << " " << k.kernelName << "(real, imag, "
                      << "_u2qParams+" << 32*u2qIdx << ", "
                      << "(1ULL<<(nqubits-" << irGenerator.vecSizeInBits << "-2))"
                      << " / " << nthreads << ", threads);\n";
                u2qIdx++; 
            } else {
                assert(false && "?");
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