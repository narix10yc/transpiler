#include "utils/PODVariant.h"
#include "utils/List.h"

#include <iostream>


int main() {
  utils::List<int> list;

  const auto printList = [&list]() {
    for (const auto i : list)
      std::cerr << i << " - ";
    std::cerr << "END\n";
  };

  list.emplace_back(1);
  printList();

  list.emplace_back(2);
  printList();

  list.emplace_back(3);
  printList();

  auto it = list.begin();
  ++it;
  list.insert(it, 4);
  list.erase(it);
  printList();

  return 0;
}