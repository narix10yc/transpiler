#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>

constexpr size_t SIZE_MB = 10240; // Test memory size in MB
constexpr size_t SIZE_B = (SIZE_MB * 1024 * 1024);

// Function to measure memory bandwidth using memcpy
double benchmark_memcpy(void* dst, void* src, size_t size_in_bytes) {
  auto start = std::chrono::high_resolution_clock::now();

  // std::memcpy(dst, src, size_in_bytes);
  std::memset(dst, 0, size_in_bytes);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  return (size_in_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed.count(); // GB/s
}


int main() {
  volatile size_t* src = (size_t*)aligned_alloc(64, SIZE_B); // Ensure alignment
  volatile size_t* dst = (size_t*)aligned_alloc(64, SIZE_B); // Ensure alignment

  std::cout << "Measuring memcpy bandwidth..." << std::endl;
  std::cout << "Memcpy speed: " << benchmark_memcpy((void*)dst, (void*)src, SIZE_B) << " GB/s" << std::endl;

  free((void*)src);
  free((void*)dst);
  return 0;
}