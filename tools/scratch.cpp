#include <iostream>
#include "utils/TaskDispatcher.h"

int main(int argc, char** argv) {
  utils::TaskDispatcher dispatcher(2);

  for (int i = 0; i < 50; i++) {
    dispatcher.enqueue([i]() {
      // std::cerr << i << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
    });
  }

  dispatcher.sync(true);
}