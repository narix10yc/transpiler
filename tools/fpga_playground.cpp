#include "saot/Fusion.h"
#include "saot/ast.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"
#include "saot/FPGAInst.h"

#include "openqasm/parser.h"

#include <cmath>

using namespace saot;
using namespace saot::fpga;

void printInstructionStatistics(const std::vector<fpga::Instruction>& insts) {
    int nNonExtMemInst = 0, nExtMemInst = 0, nSqGateInst = 0, nUpGateInst = 0;
    for (const auto& inst : insts) {
        if (inst.gInst->getKind() == GOp_SQ)
            ++nSqGateInst;
        else if (inst.gInst->getKind() == GOp_UP)
            ++nUpGateInst;
        if (!inst.mInst->isNull()) {
            if (inst.mInst->getKind() == MOp_EXT)
                ++nExtMemInst;
            else
                ++nNonExtMemInst;
        }
    }
    std::cerr << IOColor::CYAN_FG << " -- Instruction Statistics -- \n"
              << "# instructions: " << insts.size() << "\n"
              << "  - # gate instructions:   " << nSqGateInst + nUpGateInst << "\n"
              << "    - # SQ gate instructions: " << nSqGateInst << "\n"
              << "    - # UP gate instructions: " << nUpGateInst << "\n"
              << "  - # memory instructions: " << nExtMemInst + nNonExtMemInst << "\n"
              << "    - # EXT mem instructions:     " << nExtMemInst << "\n"
              << "    - # non-EXT mem instructions: " << nNonExtMemInst << "\n"
              << IOColor::RESET;

}

int main(int argc, char** argv) {
    assert(argc > 1);

    std::vector<fpga::Instruction> instructions;

    
    openqasm::Parser qasmParser(argv[1], -1);
    auto G = qasmParser.parse()->toCircuitGraph();

    // parse::Parser saotParser(argv[1]);
    // auto G = saotParser.parseQuantumCircuit().toCircuitGraph();

    // auto G = CircuitGraph::QFTCircuit(std::stoi(argv[1]));
    // auto G = CircuitGraph::ALACircuit(std::stoi(argv[1]));

    // G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";


    // instructions = fpga::genInstruction(G, instGenConfig);
    // for (const auto& i : instructions) {
    //     // i.print(std::cerr);
    //     if (!i.mInst->isNull())
    //         nMemInst++;
    //     if (!i.gInst->isNull())
    //         nGateInst++;
    //     if (!i.mInst->isNull() && i.gInst->isNull())
    //         nMemOnlyInst++;
    // }
    // std::cerr << "A total of " << instructions.size()
    //           << " (" << nMemInst << " mem, " << nGateInst << " gate) instructions\n";
    // std::cerr << "Num MemOnly instructions: " << nMemOnlyInst << "\n";
    // std::cerr << "IPC: " << (double)(nMemInst + nGateInst) / (instructions.size()) << "\n";

    applyFPGAGateFusion(FPGAFusionConfig::Default, G);

    // G.relabelBlocks();
    // G.print(std::cerr);
    std::cerr << "After fusion there are " << G.countBlocks() << " blocks\n";

    int gridSize = 4;
    // FPGAInstGenConfig instGenConfig(/* nLocalQubits = */ G.nqubits - 2 * gridSize, gridSize);
    // FPGAInstGenConfig instGenConfig(/* nLocalQubits = */ 1, gridSize);
    FPGAInstGenConfig instGenConfig(/* nLocalQubits = */ 14, gridSize);


    instructions = fpga::genInstruction(G, instGenConfig);
    printInstructionStatistics(instructions);

    double tTotal = 0.0;
    FPGACostConfig costConfig(52, 1, 1, 1, 2);
    for (const auto& inst : instructions) {
        // inst.print(std::cerr);
        tTotal += inst.cost(costConfig);
    }
    
    std::cerr << "Time taken = " << tTotal << "\n";

    return 0;
}
