$llvm_root/bin/clang -Ofast -march=native ../tmp/gen_file.ll -S -emit-llvm \
-o ../tmp/gen_file.ll.ll \
&& \
$llvm_root/bin/clang++ -Ofast -march=native ../tmp/gen_file.ll ../tmp/benchmark.cpp \
-I../include -Ltimeit -ltimeit \
-o benchmark && \
echo "Compilation finished!"