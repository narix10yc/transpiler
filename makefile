C = /opt/homebrew/Cellar/llvm/17.0.6_1/bin/clang
CXX = /opt/homebrew/Cellar/llvm/17.0.6_1/bin/clang++

INCLUDE_DIR = include
OBJ_DIR = obj
SRC_DIR = src

CFLAGS = -I$(INCLUDE_DIR)


openqasm: $(OBJ_DIR)/openqasm.o
	$(CXX) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm $(OBJ_DIR)/*

.PHONY: clean