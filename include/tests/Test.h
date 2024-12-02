
#ifndef SAOT_TESTS_TEST_H
#define SAOT_TESTS_TEST_H

#include <string>

namespace saot::test {

class TestResult {
public:
  bool success;
  std::string failureMsg;

  TestResult(bool success, const std::string& failureMsg)
    : success(success), failureMsg(failureMsg) {}

  template<typename T>
  static TestResult AssertEqual(const T& a, const T& b) {
    return TestResult(a == b, "Assert")
  }

};

template<typename T>
class AssertEqual : public TestResult {
public:
  AssertEqual(const T& a, const T& b) : success(a == b) {
    msg = (success) ? "Success" : "AssertEqual Failed";
  }
};

} // namespace saot::test

#endif // SAOT_TESTS_TEST_H