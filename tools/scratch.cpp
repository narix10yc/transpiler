#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

int main() {
  utils::TaskDispatcher dispatcher(4);
  for (int i = 0; i < 10; i++) {
    dispatcher.enqueue([i]() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distr(100, 500);
      std::cerr << "TaskA " << i << " started\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(distr(gen)));
      std::cerr << "TaskA " << i << " finished\n";
    });
  }

  dispatcher.sync();

  for (int i = 0; i < 10; i++) {
    dispatcher.enqueue([i]() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distr(100, 500);
      std::cerr << "TaskB " << i << " started\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(distr(gen)));
      std::cerr << "TaskB " << i << " finished\n";
    });
  }

  dispatcher.sync();
  return 0;
}