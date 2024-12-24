#include <iostream>

int f() {
  return 5; // Returns a temporary object
}

int main() {
  int&& a = f(); // Bind rvalue reference to the temporary returned by f()
  std::cout << "Initial value of a: " << a << std::endl; // Outputs: 5

  a = 10; // Modify the temporary through the rvalue reference
  std::cout << "Modified value of a: " << a << std::endl; // Outputs: 10

  return 0;
}