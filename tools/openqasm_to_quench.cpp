#include "openqasm/parser.h"
#include "quench/CircuitGraph.h"
#include "quench/cpu.h"

#include "llvm/Support/CommandLine.h"

#include <chrono>
#include <sstream>

using CircuitGraph = quench::circuit_graph::CircuitGraph;
using CodeGeneratorCPU = quench::cpu::CodeGeneratorCPU;
using namespace llvm;

int main(int argc, char** argv) {
    cl::opt<std::string>
    inputFilename(cl::desc("input file name"), cl::Positional, cl::Required);
    
    cl::opt<std::string>
    outputFilename("o", cl::desc("output file name"), cl::Required);

    cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));

    cl::opt<unsigned>
    VecSizeInBits("S", cl::desc("vector size in bits"), cl::Prefix, cl::init(1));

    cl::opt<unsigned>
    MaxNQubits("max_k", cl::desc("maximum number of qubits of gates"), cl::init(2));

    cl::opt<unsigned>
    NThreads("nthreads", cl::desc("number of threads"), cl::init(1));

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
    auto graph = qasmRoot->toCircuitGraph();

    tok = clock::now();

    std::cerr << get_msg_start() << "converted to CircuitGraph\n";
    graph.displayInfo(std::cerr, 2);

    tic = clock::now();
    graph.greedyGateFusion(MaxNQubits);
    tok = clock::now();

    std::cerr << get_msg_start() << "Greedy gate fusion complete\n";
    graph.displayInfo(std::cerr, 2);

    tic = clock::now();
    CodeGeneratorCPU codeGenerator(outputFilename);
    codeGenerator.generate(graph);
    tok = clock::now();

    std::cerr << get_msg_start() << "Code generation done\n";

    return 0;
}