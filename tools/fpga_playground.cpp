#include "saot/Fusion.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"
#include "saot/FPGAInst.h"

#include "openqasm/LegacyParser.h"

using namespace saot;

int main(int argc, char** argv) {
    assert(argc > 1);

    int nMemInst = 0;
    int nGateInst = 0;
    int nMemOnlyInst = 0;
    std::vector<fpga::Instruction> instructions;

    // openqasm::LegacyParser qasmLegacyParser(argv[1], -1);
    // auto G = qasmLegacyParser.parse()->toCircuitGraph();

    ast::LegacyParser saotLegacyParser(argv[1]);
    auto G = saotLegacyParser.parse().toCircuitGraph();

    G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";
    instructions = fpga::genInstruction(G, fpga::FPGAInstGenConfig::Grid2x2);
    for (const auto& i : instructions) {
        i.print(std::cerr);
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
    G.print(std::cerr);
    std::cerr << "After fusion there are " << G.countBlocks() << " blocks\n";

    instructions = fpga::genInstruction(G, fpga::FPGAInstGenConfig::Grid2x2);
    nMemInst = 0;
    nGateInst = 0;
    nMemOnlyInst = 0;
    for (const auto& i : instructions) {
        i.print(std::cerr);
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
