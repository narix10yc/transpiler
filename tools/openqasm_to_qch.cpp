#include "openqasm/parser.h"
#include "qch/ast.h"
#include "simulation/cpu.h"

using namespace simulation;


int main(int argc, char *argv[]) {

    std::string inputFilename = argv[1];

    std::string outputFilename = argv[2];

    std::cerr << "-- Input file: " << inputFilename << "\n";
    std::cerr << "-- Output file: " << outputFilename << "\n";

    openqasm::Parser parser(inputFilename, 0);

    std::string qchFileName = inputFilename + ".qch";

    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";

    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";
    
    CPUGenContext ctx {2, outputFilename};
    ctx.generate(*qchRoot);

    return 0;
}