#include "simulation/ir_generator.h"

using namespace llvm;
using namespace simulation;
using namespace quench::circuit_graph;
using GateMatrix = quench::quantum_gate::GateMatrix;
using namespace saot;

std::pair<Value*, Value*>
IRGenerator::generatePolynomial(const Polynomial& p, Value* paramArgV) {


    return { nullptr, nullptr };
}

Function* IRGenerator::generatePrepareParameter(const CircuitGraph& graph) {
    Type* scalarTy = getScalarTy();

    auto* funcTy = FunctionType::get(builder.getVoidTy(),
            { builder.getPtrTy(), builder.getPtrTy() }, false);
    Function* func = Function::Create(funcTy, Function::ExternalLinkage,
            "prepare_function", *mod);

    auto* paramArgV = func->getArg(0);
    paramArgV->setName("param.ptr");
    auto* matrixArgV = func->getArg(1);
    matrixArgV->setName("matrix.ptr");

    BasicBlock* entryBB = BasicBlock::Create(llvmContext, "entry", func);
    builder.SetInsertPoint(entryBB);

    uint64_t startIndex = 0;
    const auto allBlocks = graph.getAllBlocks();
    for (unsigned i = 0; i < allBlocks.size(); i++) {
        GateBlock* gateBlock = allBlocks[i];
        uint64_t numCompMatrixEntries = (1ULL << (2 * gateBlock->nqubits));

        const GateMatrix& gateMatrix = gateBlock->quantumGate->gateMatrix;
        if (gateMatrix.isConstantMatrix()) {
            const auto& cData = gateMatrix.cData();
            for (uint64_t d = 0; d < numCompMatrixEntries; d++) {
                std::string gepName;
                Value* matPtrV;
                // real part
                gepName = "m.block" + std::to_string(i) + ".re" + std::to_string(d);
                matPtrV = builder.CreateConstInBoundsGEP1_64(
                        scalarTy, matrixArgV, startIndex + 2*d, gepName);
                builder.CreateStore(ConstantFP::get(scalarTy, cData[d].real()), matPtrV);

                // imag part
                gepName = "m.block" + std::to_string(i) + ".im" + std::to_string(d);
                matPtrV = builder.CreateConstInBoundsGEP1_64(
                        scalarTy, matrixArgV, startIndex + 2*d + 1, gepName);
                builder.CreateStore(ConstantFP::get(scalarTy, cData[d].imag()), matPtrV);
            }
        } else {
            assert(gateMatrix.isParametrizedMatrix());
            const auto& pData = gateMatrix.pData();
            for (uint64_t d = 0; d < numCompMatrixEntries; d++) {
                auto polyV = generatePolynomial(pData[d], paramArgV);
                std::string gepName;
                Value* matPtrV;
                // real part
                gepName = "m.block" + std::to_string(i) + ".re" + std::to_string(d);
                matPtrV = builder.CreateConstInBoundsGEP1_64(
                        scalarTy, matrixArgV, startIndex + 2*d, gepName);
                builder.CreateStore(polyV.first, matPtrV);

                // imag part
                gepName = "m.block" + std::to_string(i) + ".im" + std::to_string(d);
                matPtrV = builder.CreateConstInBoundsGEP1_64(
                        scalarTy, matrixArgV, startIndex + 2*d + 1, gepName);
                builder.CreateStore(polyV.second, matPtrV);
            }
        }

        startIndex += 2 * numCompMatrixEntries;
    }
    builder.CreateRetVoid();

    return func;
}