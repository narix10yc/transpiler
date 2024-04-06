#ifndef SIMULATION_CPU_H_
#define SIMULATION_CPU_H_

#include <map>
// #include <iostream>
#include <fstream>
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

    return newName + by;
}
} // <anonymous> namespace

namespace simulation {

class CPUGenContext {
    unsigned vecSizeInBits;
    std::map<uint32_t, std::string> gateMap;
    simulation::IRGenerator irGenerator;
    std::string fileName;
    std::ofstream shellFile;
    std::ofstream cFile;
    std::ofstream incFile;
    std::ofstream irFile;
public:
    CPUGenContext(unsigned vecSizeInBits, std::string fileName)
      : vecSizeInBits(vecSizeInBits),
        gateMap(),
        irGenerator(vecSizeInBits),
        fileName(fileName) {}

    void logError(std::string msg) {}

    void generate(qch::ast::RootNode& root) {
        auto shellName = augmentFileName(fileName, "sh");
        shellFile = std::ofstream(shellName);
        std::cerr << "shell script will be written to: " << shellName << "\n";

        auto cName = augmentFileName(fileName, "c");
        cFile = std::ofstream(cName);
        std::cerr << "C script will be written to: " << cName << "\n";

        auto incName = augmentFileName(fileName, "inc");
        incFile = std::ofstream(incName);
        std::cerr << "include file will be written to: " << incName << "\n";

        auto irName = augmentFileName(fileName, "ll");
        irFile = std::ofstream(irName);
        std::cerr << "IR file will be written to: " << irName << "\n";

        root.genCPU(*this);

        shellFile.close();
        cFile.close();
        incFile.close();
        irFile.close();
    }
};

} // namespace simulation

#endif // SIMULATION_CPU_H_