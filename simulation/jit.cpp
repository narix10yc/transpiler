#include "simulation/jit.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Pass.h"

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

using namespace saot;
using namespace llvm;
using namespace simulation;
using namespace saot::jit;

JitEngine::JitEngine(IRGenerator& G) {
    InitializeNativeTarget();
    JIT = std::move(cantFail(orc::LLJITBuilder().create()));
    cantFail(JIT->addIRModule(orc::ThreadSafeModule(std::move(G._module), std::move(G._context))));
}


class DumpingObjectLinkingLayer : public orc::ObjectLinkingLayer {
public:
    // using ObjectLinkingLayer::ObjectLinkingLayer;

    void emit(std::unique_ptr<orc::MaterializationResponsibility> R, std::unique_ptr<MemoryBuffer> O) override {
        // Generate a unique file name
        std::string FileName = "dumped_object_" + std::to_string(fileIndex++) + ".o";
        std::error_code EC;
        raw_fd_ostream Out(FileName, EC, sys::fs::OF_None);

        if (EC) {
            errs() << "Error dumping object to file: " << EC.message() << "\n";
        } else {
            // Dump the object file to disk
            Out.write(reinterpret_cast<const char *>(O->getBufferStart()), O->getBufferSize());
            outs() << "Object file dumped to: " << FileName << "\n";
        }

        // Call base emit function to continue normal JIT process
        ObjectLinkingLayer::emit(std::move(R), std::move(O));
    }

private:
    static inline int fileIndex = 0; // To generate unique file names
};


void JitEngine::dumpNativeAssembly(raw_ostream& os) {
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    auto TM = cantFail(orc::JITTargetMachineBuilder(JIT->getTargetTriple()).createTargetMachine());
    auto ObjLayer = dynamic_cast<DumpingObjectLinkingLayer*>(new orc::ObjectLinkingLayer(JIT->getExecutionSession()));

    // ObjLayer->emit()



}