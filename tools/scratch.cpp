#include "utils/PODVariant.h"

#include <iostream>

int f() { return 1; }

int main() {
  using Variant = utils::PODVariant<int, double>;

  Variant v1(2);

  std::cerr << "v1.get<int>() = " << v1.get<int>() << "\n";
  std::cerr << "v1.get<double>() = " << v1.get<double>() << "\n";


  Variant v2(f());
  std::cerr << "v2.get<int>() = " << v2.get<int>() << "\n";
  std::cerr << "v2.get<double>() = " << v2.get<double>() << "\n";

  v2 = 1.1;
  std::cerr << "v2.get<int>() = " << v2.get<int>() << "\n";
  std::cerr << "v2.get<double>() = " << v2.get<double>() << "\n";

  v2.is<double>();
  return 0;
}