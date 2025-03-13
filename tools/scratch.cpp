// #include "utils/TaskDispatcher.h"
// #include "utils/StatevectorCUDA.h"
#include "utils/iocolor.h"
#include "cast/CircuitGraph.h"
#include "cast/AST.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

using namespace cast;

int main() {
  // utils::StatevectorCUDA<float> svCUDA(6);
  // svCUDA.initialize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;

  // svCUDA.randomize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;
  // svCUDA.randomize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;
  // cudaDeviceSynchronize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;

  // CircuitGraph graph;
  // CircuitGraph::QFTCircuit(32, graph);
  
  // auto qc = ast::QuantumCircuit::FromCircuitGraph(graph);
  // qc.print(std::cerr);
  

  return 0;
}