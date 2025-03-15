#include "utils/TaskDispatcher.h"
#include "simulation/StatevectorCUDA.h"
#include "utils/iocolor.h"
#include "cast/CircuitGraph.h"
#include "cast/AST.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

using namespace cast;

int main() {
  utils::StatevectorCUDA<double> svCUDA(6);
  svCUDA.initialize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;

  svCUDA.randomize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;
  // svCUDA.randomize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;
  // cudaDeviceSynchronize();
  // std::cout << "Norm: " << svCUDA.norm() << std::endl;

  

  return 0;
}