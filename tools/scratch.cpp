#include "utils/TaskDispatcher.h"
#include "utils/StatevectorCUDA.h"
#include "utils/iocolor.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

int main() {
  utils::StatevectorCUDA<double> svCUDA(6);

  svCUDA.randomize();
  return 0;
}