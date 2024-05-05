#ifndef PERFMODEL_UNIT_TEST_H_
#define PERFMODEL_UNIT_TEST_H_

#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace simulation::test {

class Test {
public:
    std::string name;
    std::vector<std::string> descs;
    std::vector<std::function<bool()>> testCases;

    void addTestCase(std::function<bool()> testCase, std::string desc="") {
        descs.push_back(desc);
        testCases.push_back(testCase);
    }

    size_t countTestCase() const { return descs.size(); }

    virtual void setup() {}
    virtual void teardown() {}
    virtual ~Test() = default;
};


class TestSuite {
private:
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string RESET = "\033[0m";
    std::vector<Test*> tests;
public:
    void addTest(Test* t) { tests.push_back(t); }

    int countTotalTestCase() const {
        int s = 0;
        for (Test* t : tests) {
            s += t->countTestCase();
        }
        return s;
    }

    void runAll() {
        int passed, failed;
        std::cerr << "Test suite starting... Total tests: " << tests.size() << 
                     "; Total runs: " << countTotalTestCase() << "\n";

        for (size_t i = 0; i < tests.size(); i++) {
            passed = 0; failed = 0;
            auto& test = tests[i];
            auto& descs = test->descs;
            auto& testCases = test->testCases;
            std::string& name = test->name;
            std::cerr << "Test " << i << ": " << name << "\n";

            test->setup();
            size_t idx = 0;
            while (idx < test->countTestCase()) {
                if (testCases[idx]()) { ++passed; }
                else { 
                    ++failed;
                    std::cerr << "  " << RED << "failed: " << descs[idx] << RESET << "\n";
                }
                ++idx;
            }
            std::cerr << "  [passed/failed]: " << "[" << passed << "/" << failed << "]\n";
            test->teardown();
            if (failed > 0) {
                std::cerr << "  " << name << ": " << RED << failed << " tests failed" << RESET << "\n";
            }
            else {
                std::cerr << "  " << name << ": " << GREEN << "all passed" << RESET << "\n";
            }
        }
        std::cerr << "Test suite finished\n";
    }
};



} // namspace simulation::test

#endif // PERFMODEL_UNIT_TEST_H_