#include "saot/Fusion.h"
#include "saot/ast.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"
#include "saot/FPGAInst.h"

#include "openqasm/parser.h"

#include <cmath>

using namespace saot;
using namespace saot::fpga;

using namespace IOColor;


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
    std::cerr << CYAN_FG << BOLD << "Num Instructions: " << insts.size() << "\n" << RESET
              << "  - # gate instructions:   " << nSqGateInst + nUpGateInst << "\n"
              << "    - # SQ gate instructions: " << nSqGateInst << "\n"
              << "    - # UP gate instructions: " << nUpGateInst << "\n"
              << "  - # memory instructions: " << nExtMemInst + nNonExtMemInst << "\n"
              << "    - # EXT mem instructions:     " << nExtMemInst << "\n"
              << "    - # non-EXT mem instructions: " << nNonExtMemInst << "\n"
              << IOColor::RESET;
}

int costKindToNumNormalizedCycle(Instruction::CostKind kind) {
    switch (kind) {
    case Instruction::CK_TwiceExtMemTime: return 84;
    case Instruction::CK_ExtMemTime: return 42;
    case Instruction::CK_NonExtMemTime: return 1;

    case Instruction::CK_GeneralSQGate: return 2;
    case Instruction::CK_RealOnlySQGate: return 1;
    case Instruction::CK_UPGate: return 1;
    default:
        assert(false && "Unreachable");
        return 0;
    }
}

int main(int argc, char** argv) {
    assert(argc > 1);

    using clock = std::chrono::high_resolution_clock;
    auto tic = clock::now();
    auto tok = clock::now();
    auto log = [&]() -> std::ostream& {
        const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count();
        return std::cerr << "-- (" << t_ms << " ms) ";
    };

    std::vector<fpga::Instruction> instructions;

    openqasm::Parser qasmParser(argv[1], -1);
    auto G = qasmParser.parse()->toCircuitGraph();

    // parse::Parser saotParser(argv[1]);
    // auto G = saotParser.parseQuantumCircuit().toCircuitGraph();

    // auto G = CircuitGraph::QFTCircuit(std::stoi(argv[1]));
    // auto G = CircuitGraph::ALACircuit(std::stoi(argv[1]));

    // G.print(std::cerr);
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";

    FPGAFusionConfig fusionConfig {
        .maxUnitaryPermutationSize = 5,
        .ignoreSingleQubitNonCompGates = true,
        .multiTraverse = true,
        .tolerances = FPGAGateCategoryTolerance::Default,
    };

    tic = clock::now();
    applyFPGAGateFusion(G, fusionConfig);
    tok = clock::now();
    log() << "Fusion complete!\n";

    std::cerr << "After fusion there are " << G.countBlocks() << " blocks\n";

    FPGAInstGenConfig instGenConfig {
        .nLocalQubits = 14,
        .gridSize = 4,
        .selectiveGenerationMode = true,
        .tolerances = FPGAGateCategoryTolerance::Default,
    };

    tic = clock::now();
    instructions = fpga::genInstruction(G, instGenConfig);
    tok = clock::now();
    log() << "Inst Gen Complete!\n";

    printInstructionStatistics(instructions);

    FPGACostConfig costConfig {
        .numLocalQubitsForTwiceExtMemOpTime = instGenConfig.nLocalQubits,
        .localQubitSignificanceForTwiceExtMemOpTime = 7
    };

    int nTwiceExtMemTime = 0, nExtMemTime = 0, nNonExtMemTime = 0;
    int nGeneralSQGate = 0, nRealOnlySQGate = 0, nUPGate = 0;

    for (const auto& inst : instructions) {
        // inst.print(std::cerr);
        auto costKind = inst.getCostKind(costConfig);
        switch (costKind) {
        case Instruction::CK_TwiceExtMemTime:
            ++nTwiceExtMemTime;
            break;
        case Instruction::CK_ExtMemTime:
            ++nExtMemTime;
            break;
        case Instruction::CK_NonExtMemTime:
            ++nNonExtMemTime;
            break;
        case Instruction::CK_GeneralSQGate:
            ++nGeneralSQGate;
            break;
        case Instruction::CK_RealOnlySQGate:
            ++nRealOnlySQGate;
            break;
        case Instruction::CK_UPGate:
            ++nUPGate;
            break;
        default:
            assert(false && "Unreachable");
            break;
        }
    }

    int tTotal = 84 * nTwiceExtMemTime +
                 42 * nExtMemTime + 
                 1 * nNonExtMemTime + 
                 2 * nGeneralSQGate + 
                 1 * nRealOnlySQGate + 
                 1 * nUPGate; 

    std::cerr << CYAN_FG << BOLD << "Num Normalized Cycles: " << tTotal << "\n" << RESET
              << "  - nTwiceExtMemTime: " << nTwiceExtMemTime << "\n"
              << "  - nExtMemTime:      " << nExtMemTime << "\n"
              << "  - nNonExtMemTime:   " << nNonExtMemTime << "\n"
              << "  - nGeneralSQGate:   " << nGeneralSQGate << "\n"
              << "  - nRealOnlySQGate:  " << nRealOnlySQGate << "\n"
              << "  - nUPGate:          " << nUPGate << "\n";


    return 0;
}
