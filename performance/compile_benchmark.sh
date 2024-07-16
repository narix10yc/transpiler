# $llvm_root/bin/clang -Ofast -march=native ../performance/gen_file.ll -S \
# -o ../performance/gen_file.ll.s \
# && \
$llvm_root/bin/clang++ -Ofast -march=native ../performance/benchmark.cpp ../performance/gen_file.ll \
-I../include -Ltimeit -ltimeit -lpthread -std=c++17 \
-o benchmark && \
echo "Compilation finished!"