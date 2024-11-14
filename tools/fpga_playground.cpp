#include "saot/Fusion.h"
#include "saot/ast.h"
#include "saot/CircuitGraph.h"
#include "saot/Parser.h"
#include "saot/FPGAInst.h"

#include "openqasm/parser.h"

#include <cmath>
#include <chrono>

using namespace saot;
using namespace saot::fpga;

using namespace IOColor;

void printInstructionStatistics(
        const std::vector<fpga::Instruction>& insts, const FPGACostConfig& costConfig) {
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

    int nTwiceExtMemTime = 0, nExtMemTime = 0, nNonExtMemTime = 0;
    int nGeneralSQGate = 0, nRealOnlySQGate = 0, nUPGate = 0;

    for (const auto& inst : insts) {
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

    std::cerr << CYAN_FG << BOLD
              << "====== Instruction Statistics: =======\n"
              << "Num Instructions: " << insts.size() << "\n" << RESET << CYAN_FG
            //   << "  - num gate instructions:   " << nSqGateInst + nUpGateInst << "\n"
            //   << "    - num SQ gate instructions: " << nSqGateInst << "\n"
            //   << "    - num UP gate instructions: " << nUpGateInst << "\n"
            //   << "  - num memory instructions: " << nExtMemInst + nNonExtMemInst << "\n"
            //   << "    - num EXT mem instructions:     " << nExtMemInst << "\n"
            //   << "    - num non-EXT mem instructions: " << nNonExtMemInst << "\n"
              << " ----------------------------------- \n"
              << BOLD << "Num Normalized Cycles: " << tTotal << "\n" << RESET << CYAN_FG
              << "  - nTwiceExtMemTime: " << nTwiceExtMemTime << "\n"
              << "  - nExtMemTime:      " << nExtMemTime << "\n"
              << "  - nNonExtMemTime:   " << nNonExtMemTime << "\n"
              << "  - nGeneralSQGate:   " << nGeneralSQGate << "\n"
              << "  - nRealOnlySQGate:  " << nRealOnlySQGate << "\n"
              << "  - nUPGate:          " << nUPGate << "\n";}

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

// This will have severe memory leak (as our CircuitGraph does not release memory)
void runExperiment(std::function<CircuitGraph()> f) {
    CircuitGraph G;
    std::vector<Instruction> instructions;

    using clock = std::chrono::high_resolution_clock;
    auto tic = clock::now();
    auto tok = clock::now();
    auto log = [&]() -> std::ostream& {
        const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tok - tic).count();
        return std::cerr << "-- (" << t_ms << " ms) ";
    };

    int nLocalQubits = 14;
    int gridSize = 4;
    FPGAFusionConfig fusionConfig {
        .maxUnitaryPermutationSize = 5,
        .ignoreSingleQubitNonCompGates = true,
        .multiTraverse = true,
        .tolerances = FPGAGateCategoryTolerance::Default,
    };

    FPGACostConfig costConfig {
        .lowestQIdxForTwiceExtTime = 7
    };
    
    FPGAInstGenConfig instGenSelectiveConfig {
        .nLocalQubits = nLocalQubits,
        .gridSize = gridSize,
        .selectiveGenerationMode = true,
        .tolerances = FPGAGateCategoryTolerance::Default,
    };

    FPGAInstGenConfig instGenNonSelectiveConfig {
        .nLocalQubits = nLocalQubits,
        .gridSize = gridSize,
        .selectiveGenerationMode = false,
        .tolerances = FPGAGateCategoryTolerance::Default,
    };

    FPGAInstGenConfig instGenBadConfig {
        .nLocalQubits = nLocalQubits,
        .gridSize = gridSize,
        .selectiveGenerationMode = false,
        .tolerances = FPGAGateCategoryTolerance::Zero,
    };

    std::cerr << YELLOW_FG << BOLD << "Test 0: Fusion  ON, InstGen  ON\n" << RESET;
    G = f();
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";
    tic = clock::now();
    applyFPGAGateFusion(G, fusionConfig);
    tok = clock::now();
    log() << "Fusion complete! " << G.countBlocks() << " blocks remain\n";

    tic = clock::now();
    instructions = fpga::genInstruction(G, instGenSelectiveConfig);
    tok = clock::now();
    log() << "Inst Gen Complete!\n";
    printInstructionStatistics(instructions, costConfig);


    std::cerr << YELLOW_FG << BOLD << "Test 1: Fusion  ON, InstGen OFF\n" << RESET;
    G = f();
    std::cerr << "Before fusion there are " << G.countBlocks() << " blocks\n";
    tic = clock::now();
    applyFPGAGateFusion(G, fusionConfig);
    tok = clock::now();
    log() << "Fusion complete! " << G.countBlocks() << " blocks remain\n";

    tic = clock::now();
    instructions = fpga::genInstruction(G, instGenNonSelectiveConfig);
    tok = clock::now();
    log() << "Inst Gen Complete!\n";
    printInstructionStatistics(instructions, costConfig);

    std::cerr << YELLOW_FG << BOLD << "Test 2: Fusion OFF, InstGen  ON\n" << RESET;
    G = f();
    tic = clock::now();
    instructions = fpga::genInstruction(G, instGenSelectiveConfig);
    tok = clock::now();
    log() << "Inst Gen Complete!\n";
    printInstructionStatistics(instructions, costConfig);


    std::cerr << YELLOW_FG << BOLD << "Test 3: Fusion OFF, InstGen OFF\n" << RESET;
    G = f();
    tic = clock::now();
    instructions = fpga::genInstruction(G, instGenNonSelectiveConfig);
    tok = clock::now();
    log() << "Inst Gen Complete!\n";
    printInstructionStatistics(instructions, costConfig);

    // std::cerr << YELLOW_FG << BOLD << "Test 4: Fusion OFF, InstGen OFF, No gate value tolerance\n" << RESET;
    // G = f();
    // tic = clock::now();
    // instructions = fpga::genInstruction(G, instGenBadConfig);
    // tok = clock::now();
    // log() << "Inst Gen Complete!\n";
    // printInstructionStatistics(instructions, costConfig);
}

int main(int argc, char** argv) {
    assert(argc > 1);

    // std::vector<fpga::Instruction> instructions;

    // openqasm::Parser qasmParser(argv[1], -1);
    // auto G = qasmParser.parse()->toCircuitGraph();

    // parse::Parser saotParser(argv[1]);
    // auto G = saotParser.parseQuantumCircuit().toCircuitGraph();

    // auto G = CircuitGraph::QFTCircuit(std::stoi(argv[1]));
    // auto G = CircuitGraph::ALACircuit(std::stoi(argv[1]));

    // G.print(std::cerr);

    // runExperiment([arg = argv[1]]() {
        // openqasm::Parser qasmParser(arg, -1);
        // return qasmParser.parse()->toCircuitGraph();
    // });

    runExperiment([arg = argv[1]]() {
        return CircuitGraph::QFTCircuit(std::stoi(arg));
    });

    return 0;
}

