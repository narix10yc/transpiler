#ifndef SAOT_TESTS_TESTKIT_H
#define SAOT_TESTS_TESTKIT_H

#include <vector>
#include <sstream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define GET_INFO() __FILE__ ":" TOSTRING(__LINE__)

#define CPU_FUNC_TYPE void(void*, uint64_t, uint64_t, const void*)

namespace saot::test {

class TestSuite {
  struct FailedInstance {
    std::string title;
    std::string info;
    std::string reason;

    FailedInstance(const std::string& title, const std::string& info,
                   const std::string& reason)
      : title(title), info(info), reason(reason) {}
  };
public:
  std::string name;
  int nTests;
  std::vector<FailedInstance> failures;

  TestSuite(const std::string& name = "Test Suite")
    : name(name), nTests(0), failures() {}

  bool displayResult() const;

  template<typename T>
  void assertEqual(
      const T& a, const T& b,
      const std::string& title, const std::string& info) {
    ++nTests;
    if (a == b)
      return;
    failures.emplace_back(title, info, "AssertEqual failed");
  }

  void assertClose(
      float a, float b,
      const std::string& title, const std::string& info, float tol=1e-4);

  void assertClose(
      double a, double b,
      const std::string& title, const std::string& info, double tol=1e-8);

  void assertAllClose(
      const std::vector<double>& aVec, const std::vector<double>& bVec,
      const std::string& title, const std::string& info, double tol=1e-8);

  void assertAllClose(
      const std::vector<float>& aVec, const std::vector<float>& bVec,
      const std::string& title, const std::string& info, float tol=1e-4);

  void assertAllClose(
      const double* aArr, const double* bArr, size_t length,
      const std::string& title, const std::string& info, double tol=1e-8);
  
  void assertAllClose(
      const float* aArr, const float* bArr, size_t length,
      const std::string& title, const std::string& info, float tol=1e-4);

};

void test_applyGate();
void test_gateMatMul();

void test_cpuH();
void test_cpuU();

inline void test_all() {
  test_applyGate();
  test_gateMatMul();
  test_cpuH();
  test_cpuU();
}

} // namespace saot::test

#endif // SAOT_TESTS_TESTKIT_H