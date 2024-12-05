#include "tests/Test.h"

using namespace saot::test;

int main() {
  TestResultRegistry registry;
  registry.assertEqual(1, 0, GET_INFO("Assert 1 == 0"));
  registry.assertClose(1.1, 1.3, GET_INFO("AssertClose 1.1 and 1.3"));

  registry.displayResult();

  return 0;
}