#include "saot/Fusion.h"
#include "saot/ast.h"
#include "saot/CircuitGraph.h"
#include "saot/NewParser.h"
#include "saot/FPGAInst.h"

#include "openqasm/parser.h"

#include <cmath>

using namespace saot;
using namespace saot::fpga;

int main(int argc, char** argv) {
    assert(argc > 1);

    int nMemInst = 0;
    int nGateInst = 0;
    int nMemOnlyInst = 0;
    std::vector<fpga::Instruction> instructions;
    FPGAInstGenConfig instGenConfig = fpga::FPGAInstGenConfig::Grid2x2;
    
    openqasm::Parser qasmParser(argv[1], -1);
    auto G = qasmParser.parse()->toCircuitGraph();

    // parse::Parser saotParser(argv[1]);
    // auto G = saotParser.parseQuantumCircuit().toCircuitGraph();

    // auto G = CircuitGraph::QFTCircuit(std::stoi(argv[1]));
    // G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";


    // instructions = fpga::genInstruction(G, instGenConfig);
    // for (const auto& i : instructions) {
    //     // i.print(std::cerr);
    //     if (!i.memInst.isNull())
    //         nMemInst++;
    //     if (!i.gateInst.isNull())
    //         nGateInst++;
    //     if (!i.memInst.isNull() && i.gateInst.isNull())
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

    instructions = fpga::genInstruction(G, instGenConfig);
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

    double tTotal = 0.0;
    // double tBaseInNanoSec = 8278.0 * std::pow(2, G.nqubits - 22);
    // double tBase = 1;
    // std::cerr << "tBase = " << std::scientific << tBaseInNanoSec << "\n";

    // FPGACostConfig costConfig(tBaseInNanoSec, tBaseInNanoSec, tBaseInNanoSec, 2 * tBaseInNanoSec);
    FPGACostConfig costConfig(1, 1, 1, 2);
    for (const auto& inst : instructions)
        tTotal += inst.cost(costConfig);
    
    std::cerr << "Time taken = " << tTotal << "\n";

    return 0;
}
