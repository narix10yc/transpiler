$llvm_root/bin/clang -Ofast -S -emit-llvm ../performance/gen_file.ll -o ../performance/gen_file.ll.ll

$llvm_root/bin/clang++ -Ofast ../performance/test_irgen.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit

$llvm_root/bin/clang++ -Ofast ../performance/test_tplt.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit

# clang++ -Ofast ../performance/test_irgen.cpp ../performance/gen_file.ll \
# -o irgen_perftest -I../include -Ltimeit -ltimeit \
# --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -mavx512f -std=c++17


# g++ -Ofast ../performance/test_tplt.cpp ../timeit/timeit.cpp \
# -o tplt_perftest -I../include -ftree-vectorize