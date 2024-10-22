#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"
#include "saot/FPGAInst.h"

#include "openqasm/parser.h"

using namespace saot;

int main(int argc, char** argv) {
    assert(argc > 1);

    int nMemInst = 0;
    int nGateInst = 0;
    int nMemOnlyInst = 0;
    std::vector<fpga::Instruction> instructions;

    openqasm::Parser qasmParser(argv[1], -1);
    auto G = qasmParser.parse()->toCircuitGraph();

    // ast::Parser saotParser(argv[1]);
    // auto G = saotParser.parse().toCircuitGraph();

    // G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";
    instructions = fpga::genInstruction(G, fpga::FPGAInstGenConfig::Grid4x4);
    for (const auto& i : instructions) {
        // i.print(std::cerr);
        if (!i.memInst.isNull())
            nMemInst++;
        if (!i.gateInst.isNull())
            nGateInst++;
        if (!i.memInst.isNull() && i.gateInst.isNull())
            nMemOnlyInst++;
    }
    std::cerr << "A total of " << instructions.size()
              << " (" << nMemInst << " mem, " << nGateInst << " gate) instructions\n";
    std::cerr << "Num MemOnly instructions: " << nMemOnlyInst << "\n";
    std::cerr << "IPC: " << (double)(nMemInst + nGateInst) / (instructions.size()) << "\n";

    applyFPGAGateFusion(FPGAFusionConfig::Default, G);

    G.relabelBlocks();
    // G.print(std::cerr);
    std::cerr << "After fusion there are " << G.countBlocks() << " blocks\n";

    instructions = fpga::genInstruction(G, fpga::FPGAInstGenConfig::Grid4x4);
    nMemInst = 0;
    nGateInst = 0;
    nMemOnlyInst = 0;
    for (const auto& i : instructions) {
        // i.print(std::cerr);
        if (!i.memInst.isNull())
            nMemInst++;
        if (!i.gateInst.isNull())
            nGateInst++;
        if (!i.memInst.isNull() && i.gateInst.isNull())
            nMemOnlyInst++;
    }
    std::cerr << "A total of " << instructions.size()
              << " (" << nMemInst << " mem, " << nGateInst << " gate) instructions\n";
    std::cerr << "Num MemOnly instructions: " << nMemOnlyInst << "\n";
    std::cerr << "IPC: " << (double)(nMemInst + nGateInst) / (instructions.size()) << "\n";


    return 0;
}
