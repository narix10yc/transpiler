#include <iostream>
#include <list>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define PRINT_SIZE_OF(X) std::cout << "sizeof(" TOSTRING(X) ") = " << sizeof(X) << std::endl;

int main() {
  PRINT_SIZE_OF(std::function<void()>);
  return 0;
}