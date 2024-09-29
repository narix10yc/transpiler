#include "simulation/jit.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"

using namespace saot;
using namespace llvm;
using namespace simulation;
using namespace saot::jit;

JitEngine::JitEngine(std::unique_ptr<IRGenerator> G) {
    InitializeNativeTarget();
    JIT = std::move(cantFail(orc::LLJITBuilder().create()));
    cantFail(JIT->addIRModule(orc::ThreadSafeModule(std::move(G->_module), std::move(G->_context))));
}

void JitEngine::dumpNativeAssembly(raw_ostream& os) {
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // std::string TargetTriple = sys::getDefaultTargetTriple();
    // std::string Error;
    // const Target *Target = TargetRegistry::lookupTarget(TargetTriple, Error);
    
    // TargetOptions Options;
    // auto TargetMachine = Target->createTargetMachine(TargetTriple, "generic", "", Options, None);

    // M->setDataLayout(TargetMachine->createDataLayout());

    // // Step 6: Set up pass manager to emit assembly
    // legacy::PassManager PM;
    // if (EC) {
    //     errs() << "Error opening output.s: " << EC.message() << "\n";
    //     return 1;
    // }

    // TargetMachine->addPassesToEmitFile(PM, OS, nullptr, CodeGenFileType::CGFT_AssemblyFile);
    // PM.run(*M);

    // outs() << "Assembly file successfully written to output.s\n";
}