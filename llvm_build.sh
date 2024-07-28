#! /bin/shell

cmake -S llvm-project-17.0.6.src/llvm -G Ninja \
-B llvm-build --install-prefix=$pwd/llvm-install \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_PROJECTS="clang" \
-DLLVM_ENABLE_RUNTIMES="libcxx" \
-DLLVM_TARGETS_TO_BUILD="X86" \
-DLLVM_ENABLE_RTTI=ON
