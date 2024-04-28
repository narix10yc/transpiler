$llvm_root/bin/clang++ -O3 ../performance/test.cpp ../performance/gen_file.ll \
-o irgen_perftest -I../include -Ltimeit -ltimeit

$llvm_root/bin/clang++ -O3 ../performance/test_tplt.cpp \
-o tplt_perftest -I../include -Ltimeit -ltimeit