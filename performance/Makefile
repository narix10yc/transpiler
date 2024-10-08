# Compiler and flags
CLANG = $(llvm_root)/bin/clang
CLANGXX = $(llvm_root)/bin/clang++

BUILD_DIR = $(CURDIR)/../build
INCLUDE_DIR = $(CURDIR)/../include

CFLAGS = -std=c++17 -I$(INCLUDE_DIR) -Ofast -march=native --gcc-install-dir=$(SELECT_GCC_TOOLCHAIN)
LDFLAGS = -L$(BUILD_DIR)/timeit -ltimeit -lpthread

# Directories
# IR_DIR = $(CURDIR)/gen_file_ir
IR_DIR = $(CURDIR)
OBJ_DIR = $(CURDIR)/obj
SRC_FILE = $(CURDIR)/benchmark.cpp
HEADER_FILE =$(wildcard $(CURDIR)/*.h)

# Files
IR_FILES = $(wildcard $(IR_DIR)/*.ll)
OBJ_FILES = $(patsubst $(IR_DIR)/%.ll, $(OBJ_DIR)/%.ll.o, $(IR_FILES))

# Default target
all: $(OBJ_DIR) benchmark single_gate

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile LLVM IR files to object files
$(OBJ_DIR)/%.ll.o: $(IR_DIR)/%.ll | $(OBJ_DIR)
	$(CLANG) -c -Ofast -march=native -o $@ $<

fidelity: $(CURDIR)/gen_file.ll $(CURDIR)/fidelity.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/fidelity
	
benchmark: $(CURDIR)/gen_file.ll $(CURDIR)/benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/benchmark

gate_benchmark : $(CURDIR)/gen_file.ll $(CURDIR)/gate_benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/gate_benchmark

full_gate_bcmk : $(CURDIR)/gen_file.ll $(CURDIR)/full_gate_bcmk.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/full_gate_bcmk

single_gate : $(CURDIR)/gen_file.ll $(CURDIR)/single_gate.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/single_gate

correctness: $(CURDIR)/gen_file.ll $(CURDIR)/correctness.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -L$(BUILD_DIR)/saot -lsaot -o $(BUILD_DIR)/correctness

ir_benchmark : ir_benchmark_gcc ir_benchmark_clang

ir_benchmark_gcc: $(OBJ_FILES) $(CURDIR)/ir_benchmark.cpp
	g++ $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/ir_benchmark_gcc

ir_benchmark_clang: $(OBJ_FILES) $(CURDIR)/ir_benchmark.cpp
	$(CLANGXX) $^ $(LDFLAGS) $(CFLAGS) -o $(BUILD_DIR)/ir_benchmark_clang

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o

# Phony targets
.PHONY: all clean benchmark ir_benchmark
