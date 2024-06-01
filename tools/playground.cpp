#include "openqasm/parser.h"
#include "simulation/cpu.h"
#include "simulation/transpiler.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>
#include <fstream>

using namespace simulation;
using namespace simulation::transpile;
using namespace llvm;

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional, cl::Required);
    
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::Required);

    cl::ParseCommandLineOptions(argc, argv);

    std::cerr << "-- Input file: " << inputFilename << "\n";
    std::cerr << "-- Output file: " << outputFilename << "\n";

    openqasm::Parser parser(inputFilename, 0);


    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";
    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";
    auto graph = CircuitGraph::FromQch(*qchRoot);

    std::ofstream file(outputFilename + ".txt");
    graph.getTile().toTikZ(file);

    graph.transpileForCPU();

    file = std::ofstream {outputFilename + "transpiled.txt"};
    graph.getTile().toTikZ(file);

    return 0;
}