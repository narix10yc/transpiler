# Compiler and flags
CLANG = $(llvm_root)/bin/clang
CLANGXX = $(llvm_root)/bin/clang++

BUILD_DIR = $(CURDIR)/../build
INCLUDE_DIR = $(CURDIR)/../include

CFLAGS = -std=c++17 -I$(INCLUDE_DIR) -Ofast -march=native
LDFLAGS = -L$(BUILD_DIR)/timeit -ltimeit -lpthread

# Directories
IR_DIR = $(CURDIR)
OBJ_DIR = $(CURDIR)/obj
SRC_FILE = $(CURDIR)/benchmark.cpp
HEADER_FILE =$(wildcard $(CURDIR)/*.h)

# Files
IR_FILES = $(wildcard $(IR_DIR)/*.ll)
OBJ_FILES = $(patsubst $(IR_DIR)/%.ll, $(OBJ_DIR)/%.ll.o, $(IR_FILES))
SRC_OBJ = $(OBJ_DIR)/benchmark.cpp.o

# Default target
all: $(OBJ_DIR) benchmark single_gate

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile LLVM IR files to object files
$(OBJ_DIR)/%.ll.o: $(IR_DIR)/%.ll | $(OBJ_DIR)
	$(CLANG) -c -Ofast -march=native -o $@ $<

$(SRC_OBJ): $(SRC_FILE) $(HEADER_FILE) | $(OBJ_DIR)
	$(CLANGXX) $(CFLAGS) -c -o $@ $(SRC_FILE)

benchmark: $(OBJ_FILES) $(SRC_OBJ)
	$(CLANGXX) $^ $(LDFLAGS) -o $(BUILD_DIR)/benchmark

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
