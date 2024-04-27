#ifndef PERFMODEL_UNIT_TEST_H_
#define PERFMODEL_UNIT_TEST_H_

#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace simulation::test {

class Test {
    std::vector<std::string> desc;
    std::vector<std::function<bool()>> testCases;

public:
    std::string name;
    std::function<void()> setup = [](){};
    std::function<void()> teardown = [](){};

    Test() {}
    Test(std::string name) : name(name) {}

    void addTestCase(std::function<bool()> testCase, std::string description="") {
        desc.push_back(description);
        testCases.push_back(testCase);
    }

    size_t countTestCase() const { return desc.size(); }

    const std::vector<std::function<bool()>>& getTestCases() { return testCases; }
    const std::vector<std::string>& getDescs() { return desc; }
};


class TestSuite {
private:
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string RESET = "\033[0m";
    std::vector<Test*> tests;
public:
    void addTest(Test* t) { tests.push_back(t); }

    int testRunCount() const {
        int s = 0;
        for (Test* t : tests) {
            s += t->getTestCases().size();
        }
        return s;
    }

    void runAll() {
        int passed, failed;
        std::cerr << "Test suite starting... Total tests: " << tests.size() << 
                     "; Total runs: " << testRunCount() << "\n";

        for (size_t i = 0; i < tests.size(); i++) {
            std::string& name = tests[i]->name;

            passed = 0; failed = 0;
            std::cerr << "Test " << i << ": " << tests[i]->name << "\n";

            auto& test = *tests[i];
            auto& testCases = test.getTestCases();
            auto& descs = test.getDescs();
            test.setup();
            size_t idx = 0;
            while (idx < test.countTestCase()) {
                if (testCases[idx]()) { ++passed; }
                else { 
                    ++failed;
                    std::cerr << "  " << RED << "failed: " << descs[idx] << RESET << "\n";
                }
                ++idx;
            }
            std::cerr << "  [passed/failed]: " << "[" << passed << "/" << failed << "]\n";
            tests[i]->teardown();
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