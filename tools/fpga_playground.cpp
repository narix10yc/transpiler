#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/parser.h"

using namespace saot;

int main(int argc, char** argv) {
    assert(argc > 1);

    ast::Parser parser(argv[1]);

    auto qc = parser.parse();

    auto G = qc.toCircuitGraph();

    G.print(std::cerr);


    return 0;
}
