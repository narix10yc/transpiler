#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"

#include "openqasm/parser.h"

using namespace saot;

int main(int argc, char** argv) {
    assert(argc > 1);

    // openqasm::Parser qasmParser(argv[1], -1);
    // auto G = qasmParser.parse()->toCircuitGraph();

    ast::Parser saotParser(argv[1]);
    auto G = saotParser.parse().toCircuitGraph();

    // G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";

    applyFPGAGateFusion(FPGAFusionConfig::Default, G);

    G.relabelBlocks();
    G.print(std::cerr);
    std::cerr << "After fusion there are " << G.countBlocks() << " blocks\n";

    return 0;
}
