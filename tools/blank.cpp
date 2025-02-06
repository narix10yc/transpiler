#include <iostream>
#include <list>

int main() {
  std::cout << "Size of std::list<int>::iterator: "
            << sizeof(std::list<int>::iterator) << " bytes\n";
  std::cout << "Size of a pointer: " << sizeof(void*) << " bytes\n";
  return 0;
}