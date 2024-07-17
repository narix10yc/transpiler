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

# Files
IR_FILES = $(wildcard $(IR_DIR)/*.ll)
OBJ_FILES = $(patsubst $(IR_DIR)/%.ll, $(OBJ_DIR)/%.ll.o, $(IR_FILES))
SRC_OBJ = $(OBJ_DIR)/benchmark.cpp.o
EXEC = $(BUILD_DIR)/benchmark

# Default target
all: $(OBJ_DIR) benchmark

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	
benchmark : $(EXEC)

# Compile LLVM IR files to object files
$(OBJ_DIR)/%.ll.o: $(IR_DIR)/%.ll | $(OBJ_DIR)
	$(CLANG) -c -Ofast -march=native -o $@ $<

$(SRC_OBJ): $(SRC_FILE) | $(OBJ_DIR)
	$(CLANGXX) $(CFLAGS) -c -o $@ $<

# Link all object files
$(EXEC): $(OBJ_FILES) $(SRC_OBJ)
	$(CLANGXX) $^ $(LDFLAGS) -o $@ 

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o

# Phony targets
.PHONY: all clean benchmark
