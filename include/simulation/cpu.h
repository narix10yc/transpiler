#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>

#include "simulation/ir_generator.h"
#include "qch/ast.h"

namespace simulation {

class CPUGenContext {
    simulation::IRGenerator irGenerator;
    std::string fileName;
    std::error_code EC;

public:
    unsigned gateCount;
    std::map<uint32_t, std::string> u3GateMap;
    std::map<std::string, std::string> u2qGateMap;
    std::stringstream shellStream;
    std::stringstream declStream;
    std::stringstream u3ParamStream;
    std::stringstream u2qParamStream;
    std::stringstream kernelStream;
    std::stringstream irStream;
    unsigned vecSizeInBits;
    unsigned nqubits;

    CPUGenContext(unsigned vecSizeInBits, std::string fileName)
        : irGenerator(vecSizeInBits),
          fileName(fileName),
          vecSizeInBits(vecSizeInBits) {}
    
    void setRealTy(ir::RealTy ty) { irGenerator.setRealTy(ty); }
    void setAmpFormat(ir::AmpFormat format) { irGenerator.setAmpFormat(format); }

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

    void logError(std::string msg) {}

    simulation::IRGenerator& getGenerator() { return irGenerator; }

    void generate(qch::ast::RootNode& root) {
        std::error_code EC;
        auto shellName = fileName + ".sh";
        auto shellFile = llvm::raw_fd_ostream(shellName, EC);
        std::cerr << "shell script will be written to: " << shellName << "\n";

        auto hName = fileName + ".h";
        auto hFile = llvm::raw_fd_ostream(hName, EC);
        std::cerr << "header file will be written to: " << hName << "\n";

        auto irName = fileName + ".ll";
        auto irFile = llvm::raw_fd_ostream(irName, EC);
        std::cerr << "IR file will be written to: " << irName << "\n";

        std::string typeStr =
            (getRealTy() == ir::RealTy::Double) ? "double" : "float";

        hFile << "#include <cstdint>\n\n";
        
        u3ParamStream << "static const " << typeStr << " _u3Param[] = {\n";
        u2qParamStream << "static const " << typeStr << " _u2qParam[] = {\n";

        kernelStream << "void simulate_circuit(";
        if (getAmpFormat() == ir::AmpFormat::Separate)
            kernelStream << typeStr << " *real, " << typeStr << " *imag";
        else
            kernelStream << typeStr << " *data";
        kernelStream << ", uint64_t, uint64_t, const " << typeStr << "*) {\n";

        declStream << "extern \"C\" {\n";

        root.genCPU(*this);

        u3ParamStream << "};\n";
        u2qParamStream << "};\n";
        declStream << "}";
        kernelStream << "}";

        shellFile << shellStream.str();
        hFile << declStream.str() << "\n\n" 
              << u3ParamStream.str()  << "\n"
              << u2qParamStream.str() << "\n"
              << kernelStream.str();
        irGenerator.getModule().print(irFile, nullptr);

        shellFile.close();
        hFile.close();
        irFile.close();
    }
};

} // namespace simulation

#endif // SIMULATION_CPU_H_