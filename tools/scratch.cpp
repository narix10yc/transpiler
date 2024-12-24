#include <iostream>
#include <typeinfo>

void printNames() {}

template<typename First, typename... Args>
void printNames(First first, Args... args) {
  if constexpr (sizeof...(Args) == 0) {
    std::cerr << typeid(first).name() << "]";
    return;
  }
  std::cerr << typeid(first).name() << ", ";
  printNames(std::forward<Args>(args)...);
}

template<typename... Args>
void printNamesEntry(Args... args) {
  std::cerr << "[";
  printNames(std::forward<Args>(args)...);
}

template<typename... Args>
void printNamesNice(Args&&... args) {
// ((std::cerr << typeid(args).name() << (sizeof...(Args) == 1 ? "" : ", ")), ...) << '\n';
  ((std::cerr << "[") << ... << (typeid(args).name()));
}

int main() {
  printNamesNice(1, 2ULL, 3.0f);

  return 0;
}