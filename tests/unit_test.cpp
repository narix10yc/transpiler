#include "unit_test.h"
#include "simulation/tplt.h"
#include "simulation/statevector.h"

#include <cmath>

#define INV_SQRT2 (0.7071067811865475727373109)

using namespace simulation;
using namespace simulation::sv;
using namespace simulation::tplt;
using namespace simulation::test;


bool is_close(double a, double b) {
    return abs(a - b) < 1e-8;
}


template<typename real_ty>
class TestH : public Test {
    Statevector<real_ty> sv1, sv2;
    ComplexMatrix2<real_ty> mat;
public:
    TestH(unsigned nqubits=12)
        : sv1(nqubits, true), sv2(nqubits, true) {
        name = "test H";
        mat = {{INV_SQRT2,INV_SQRT2,INV_SQRT2,-INV_SQRT2}, {0,0,0,0}};
        addTestCase([&]() -> bool {
            applySingleQubit<real_ty>(sv1.real, sv1.imag, mat, nqubits, 0);
            if (!is_close(sv1.normSquared(), 1.0))
                return false;
            return true;
            
        }, "test gate H");
    }
};


int main() {
    auto testSuite = TestSuite();

    auto t1 = TestH<float>();
    auto t2 = TestH<double>();

    testSuite.addTest(&t1);
    testSuite.addTest(&t2);

    testSuite.runAll();
    return 0;
}