#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <map>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "simulation/irGen.h"
#include "qch/ast.h"

namespace {
std::string augmentFileName(std::string& fileName, std::string by) {
    auto lastDotPos = fileName.find_last_of(".");
    std::string newName;
    if (lastDotPos == std::string::npos) 
        newName = fileName;
    else 
        newName = fileName.substr(0, lastDotPos);

    return newName + "." + by;
}
} // <anonymous> namespace

namespace simulation {

class CPUGenContext {
    std::map<uint32_t, std::string> gateMap;
    simulation::IRGenerator irGenerator;
    std::string fileName;
    std::error_code EC;
public:
    unsigned gateCount;
    std::stringstream shellStream;
    std::stringstream cStream;
    std::stringstream incStream;
    std::stringstream irStream;
    unsigned vecSizeInBits;
    unsigned nqubits;

    CPUGenContext(unsigned vecSizeInBits, std::string fileName)
      : vecSizeInBits(vecSizeInBits),
        gateMap(),
        gateCount(0),
        irGenerator(vecSizeInBits),
        fileName(fileName) {}

    void logError(std::string msg) {}

    simulation::IRGenerator& getGenerator() { return irGenerator; }

    void generate(qch::ast::RootNode& root) {
        std::error_code EC;
        auto shellName = augmentFileName(fileName, "sh");
        auto shellFile = llvm::raw_fd_ostream(shellName, EC);
        std::cerr << "shell script will be written to: " << shellName << "\n";

        auto cName = augmentFileName(fileName, "c");
        auto cFile = llvm::raw_fd_ostream(cName, EC);
        std::cerr << "C script will be written to: " << cName << "\n";

        auto incName = augmentFileName(fileName, "inc");
        auto incFile = llvm::raw_fd_ostream(incName, EC);
        std::cerr << "include file will be written to: " << incName << "\n";

        auto irName = augmentFileName(fileName, "ll");
        auto irFile = llvm::raw_fd_ostream(irName, EC);
        std::cerr << "IR file will be written to: " << irName << "\n";

        cStream << std::setprecision(16);
        cStream << "#include \"" << incName << "\"\n\n";
        cStream << "void simulate_circuit(double* real, double* imag) {\n";

        incStream << "#include <stdint.h>\n\n";
        incStream << "typedef double v8double __attribute__((vector_size(64)));\n\n";

        root.genCPU(*this);

        cStream << "}";

        shellFile << shellStream.str();
        cFile << cStream.str();
        incFile << incStream.str();
        irGenerator.getModule().print(irFile, nullptr);

        shellFile.close();
        cFile.close();
        incFile.close();
        irFile.close();
    }
};

} // namespace simulation

#endif // SIMULATION_CPU_H_