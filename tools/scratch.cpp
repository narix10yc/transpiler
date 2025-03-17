#include "cast/Parser.h"
// #include "cast/QuantumCircuit.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <span>

using namespace cast;

int main(int argc, char** argv) {
  cast::Parser parser(argv[1]);

  auto qc = parser.parseQuantumCircuit();
  qc.print(std::cerr) << "\n";


  return 0;
}