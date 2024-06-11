#include "quench/parser.h"
#include "quench/GateMatrix.h"

using namespace quench::ast;
using namespace quench::cas;

int main() {
    Parser parser("../examples/simple.qch");
    auto root = parser.parse();

    return 0;
}