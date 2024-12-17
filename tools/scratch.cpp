#include "saot/Parser.h"

using namespace saot;

int main() {
  Parser parser("../examples/parse/p1.qch");
  auto qc = parser.parseQuantumCircuit();
  qc.print(std::cerr << "== Recovered: == \n") << "\n";
  return 0;
}