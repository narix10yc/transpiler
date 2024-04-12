#include "openqasm/parser.h"
#include "qch/ast.h"
#include "simulation/cpu.h"

using namespace simulation;


int main(int argc, char *argv[]) {

    std::string inputFilename = argv[1];

    std::cerr << "-- Input file: " << inputFilename << std::endl;

    openqasm::Parser parser(inputFilename, 0);

    std::string qchFileName = augmentFileName(inputFilename, ".qch");

    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";

    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";

    std::ofstream f(qchFileName);
    std::cerr << "-- opened " << qchFileName << "\n";

    qchRoot->print(f);

    std::cerr << "qch file written at " << qchFileName << std::endl;
    f.close();
    
    CPUGenContext ctx {2, "gen_file"};
    ctx.generate(*qchRoot);

    return 0;
}