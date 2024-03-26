C = /opt/homebrew/Cellar/llvm/17.0.6_1/bin/clang
CXX = /opt/homebrew/Cellar/llvm/17.0.6_1/bin/clang++

INCLUDE_DIR = include
OBJ_DIR = obj
SRC_DIR = src

CXXFLAGS = -I$(INCLUDE_DIR)

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

all: $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o openqasm.exe

# openqasm: $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(TARGETS)))
# $(CXX) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm $(OBJ_DIR)/*

.PHONY: clean