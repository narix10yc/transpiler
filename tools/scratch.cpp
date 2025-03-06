#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

int main() {
  std::cerr << "sizeof unique ptr " << sizeof(std::unique_ptr<int>) << "\n";
  return 0;
}