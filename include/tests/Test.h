
#ifndef SAOT_TESTS_TEST_H
#define SAOT_TESTS_TEST_H

#include "utils/iocolor.h"

#include <iostream>
#include <vector>
#include <sstream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define GET_INFO(TITLE) TITLE, __FILE__ ":" TOSTRING(__LINE__)

namespace saot::test {

class TestResultRegistry {
  struct FailedInstance {
    std::string title;
    std::string info;
    std::string reason;

    FailedInstance(const std::string& title, const std::string& info,
                   const std::string& reason)
      : title(title), info(info), reason(reason) {}
  };
public:
  std::vector<FailedInstance> failures;

  TestResultRegistry() : failures() {}

  void displayResult() const {
    if (failures.empty()) {
      std::cerr << IOColor::GREEN_FG << "All tests passed!\n" << IOColor::RESET;
      return;
    }

    unsigned n = failures.size();
    std::cerr << BOLDRED(n << " test" << ((n > 1) ? "s" : "") << " failed:\n");
    for (unsigned i = 0; i < n; i++) {
      std::cerr << RED(
        i << ": '" << failures[i].title << "' at " << failures[i].info
          << "\n  Reason: " << failures[i].reason << "\n");
    }
  }

  template<typename T>
  void assertEqual(const T& a, const T& b,
                   const std::string& title, const std::string& info) {
    if (a == b)
      return;
    failures.emplace_back(title, info, "AssertEqual failed");
  }

  void assertClose(float a, float b,
                   const std::string& title, const std::string& info,
                   float tol=1e-4) {
    float diff = std::abs(a - b);
    if (diff <= tol)
      return;
    
    std::stringstream ss;
    ss << "LHS " << a << " and RHS " << b
       << " (Diff = " << diff << ") > (Tolerance " << tol << ")";
    failures.emplace_back(title, info, ss.str());
  }

  void assertClose(double a, double b,
                   const std::string& title, const std::string& info,
                   double tol=1e-8) {
    double diff = std::abs(a - b);
    if (diff <= tol)
      return;
    
    std::stringstream ss;
    ss << "LHS " << a << ", RHS " << b
       << ", (Diff = " << diff << ") > (Tolerance = " << tol << ")";
    failures.emplace_back(title, info, ss.str());
  }

};


} // namespace saot::test

#endif // SAOT_TESTS_TEST_H