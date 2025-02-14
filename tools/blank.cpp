#include <iostream>
#include <chrono>
#include <thread>
#include <cassert>

void displayProgressBar(float progress, int barWidth = 50) {
  // Clamp progress between 0 and 1
  assert(barWidth > 0);
  if (progress < 0.0f) progress = 0.0f;
  if (progress > 1.0f) progress = 1.0f;

  // Print the progress bar
  std::cout.put('[');
  int i = 0;
  while (i < barWidth * progress) {
    std::cout.put('=');
    ++i;
  }
  while (i < barWidth) {
    std::cout.put(' ');
    ++i;
  }

  std::cout << "] " << static_cast<int>(progress * 100.0f) << " %\r";
  std::cout.flush();
}

int main() {
  const int totalSteps = 100;

  for (int i = 0; i <= totalSteps; ++i) {
    float progress = static_cast<float>(i) / totalSteps;
    displayProgressBar(progress);

    // Simulate work (e.g., computations or file processing)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Print a newline after the progress bar is complete
  std::cout << std::endl;

  return 0;
}