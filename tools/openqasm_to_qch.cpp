#include "openqasm/parser.h"
#include "simulation/cpu.h"
#include "simulation/transpiler.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>

using namespace simulation;
using namespace simulation::transpile;
using namespace llvm;

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional, cl::Required);
    
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::Required);

    cl::ParseCommandLineOptions(argc, argv);

    using clock = std::chrono::high_resolution_clock;
    auto tic = clock::now();
    auto tok = clock::now();
    auto get_msg_start = [&]() -> std::string {
        std::stringstream ss;
        ss << std::setprecision(2) << "-- ("
           << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count())
           << " ms) ";

        return ss.str();
    };

    std::cerr << "-- Input file: " << inputFilename << "\n";
    std::cerr << "-- Output file: " << outputFilename << "\n";

    openqasm::Parser parser(inputFilename, 0);

    tic = clock::now();
    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";
    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";
    auto graph = CircuitGraph::FromQch(*qchRoot);
    tok = clock::now();

    std::cerr << get_msg_start() << "converted to CircuitGraph with "
              << graph.allNodes.size() << " nodes\n";

    tic = clock::now();
    graph.transpileForCPU();
    tok = clock::now();

    std::cerr << get_msg_start() << "transpiled for CPU\n";

    tic = clock::now();
    auto transpiledRoot = graph.toQch();
    tok = clock::now();

    std::cerr << get_msg_start() << "converted back to qch AST\n";

    tic = clock::now();
    CPUGenContext ctx {1, outputFilename};
    ctx.generate(transpiledRoot);
    tok = clock::now();

    std::cerr << get_msg_start() << "generated files\n";

    return 0;
}