#include <iostream>
#include "utils/TaskDispatcher.h"

int main(int argc, char** argv) {
  utils::TaskDispatcher dispatcher(2);

  for (int i = 0; i < 5; i++) {
    dispatcher.enqueue([i]() {
      std::cerr << i << "\n";
    });
  }

  dispatcher.sync();
}