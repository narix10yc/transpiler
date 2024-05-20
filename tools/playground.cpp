#include "openqasm/parser.h"
#include "simulation/cpu.h"
#include "simulation/transpiler.h"

using namespace simulation;
using namespace simulation::transpile;


int main(int argc, char *argv[]) {
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    std::cerr << "-- Input file: " << inputFilename << "\n";
    std::cerr << "-- Output file: " << outputFilename << "\n";

    openqasm::Parser parser(inputFilename, 1);

    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";

    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";
    
    auto graph = CircuitGraph::FromQch(*qchRoot);
    
    std::cerr << "-- converted to CircuitGraph with "
              << graph.allNodes.size() << " nodes\n";

    graph.transpileForCPU();

    std::cerr << "-- transpiled for CPU\n";

    auto transpiledRoot = graph.toQch();

    std::cerr << "-- converted back to qch AST\n";

    CPUGenContext ctx {1, outputFilename};

    ctx.generate(transpiledRoot);

    return 0;
}