#include "quench/parser.h"

using namespace quench::ast;

int main() {
    Parser parser("../examples/simple.qch");
    auto root = parser.parse();
    
    return 0;
}