$llvm_root/bin/clang -Ofast -S -emit-llvm ../performance/gen_file.ll -o ../performance/gen_file.ll.ll

$llvm_root/bin/clang -Ofast -S ../performance/gen_file.ll -o ../performance/gen_file.ll.asm 

# u3 gate 
$llvm_root/bin/clang++ -Ofast ../performance/bm_irgen.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit \
&& \
$llvm_root/bin/clang++ -Ofast -march=native ../performance/bm_tplt.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit


clang++ -Ofast ../performance/bm_irgen.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit \
--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -std=c++17 -march=native \
&& \
g++ -Ofast ../performance/bm_tplt.cpp ../timeit/timeit.cpp \
-o tplt_perftest -I../include -march=native


# u2q gate 
$llvm_root/bin/clang++ -Ofast ../performance/bm_irgen_u2q.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit \
&&
$llvm_root/bin/clang++ -Ofast -march=native ../performance/bm_tplt_u2q.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit

# circuit
$llvm_root/bin/clang++ -O3 ../performance/bm_irgen_circuit.cpp \
../performance/gen_file.ll -I../include -Ltimeit -ltimeit -o circuit