#include <iostream>
#include <vector>
#include "utils/PODVector.h"

struct MyStruct {
  int a;
  void* p;
};


int main() {
  std::vector<MyStruct> v;
  // utils::PODVector<MyStruct> v;
  v.resize(2);
  std::cerr << v[0].a << "\n";

  return 0;
}