#include "quench/parser.h"
#include "quench/GateMatrix.h"

using namespace quench::ast;
using namespace quench::cas;

int main() {
    Parser parser("../examples/simple.qch");
    auto root = parser.parse();

    std::cerr << std::stoi("3.4.5.01");

    return 0;
}