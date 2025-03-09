#include "utils/TaskDispatcher.h"
#include "utils/StatevectorCUDA.h"
#include "utils/iocolor.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

int main() {
  utils::StatevectorCUDA<float> svCUDA(6);
  svCUDA.initialize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;

  svCUDA.randomize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;
  svCUDA.randomize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;
  cudaDeviceSynchronize();
  std::cout << "Norm: " << svCUDA.norm() << std::endl;

  return 0;
}