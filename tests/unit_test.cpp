#include "unit_test.h"
#include "simulation/tplt.h"
#include "simulation/statevector.h"

#include <cmath>

#define INV_SQRT2 (0.7071067811865475727373109)

typedef struct { double data[8]; } v8double;
typedef struct { float data[8]; } v8float;

extern "C" {
    void u3_f64_alt_0200ffff(double*, uint64_t, uint64_t, void*);
    void u3_f64_alt_0000ffff(double*, uint64_t, uint64_t, void*);
    void u3_f64_sep_0200ffff(double*, double*, uint64_t, uint64_t, void*);
}

using namespace simulation;
using namespace simulation::sv;
using namespace simulation::tplt;
using namespace simulation::test;


bool is_close(double a, double b, double thres=1e-8) {
    return abs(a - b) < thres;
}

class TestH : public Test {
    StatevectorSep<double> sv1, sv2;
    ComplexMatrix2<double> mat;
    double m[8] = {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2, 0, 0, 0, 0};
public:
    TestH(unsigned nqubits=12)
        : sv1(nqubits, true), sv2(nqubits, false) {
        name = "test H";
        mat = {{INV_SQRT2,INV_SQRT2,INV_SQRT2,-INV_SQRT2}, {0,0,0,0}};

        addTestCase([&]() -> bool {
            applySingleQubit<double>(sv1.real, sv1.imag, mat, sv1.nqubits, 0);
            return is_close(sv1.normSquared(), 1.0);
        }, "test gate H norm (k=0), standard method");

        addTestCase([&]() -> bool {
            sv2 = sv1;
            applySingleQubit<double>(sv1.real, sv1.imag, mat, sv1.nqubits, 2);
            u3_f64_sep_0200ffff(sv2.real, sv2.imag, 0, 1<<(sv2.nqubits-3), m);
            return is_close(fidelity(sv1, sv2), 1);
        }, "test gate H equal (k=2), standard vs sep");
    }
};

class TestSepAlt : public Test {
    StatevectorSep<double> svSep;
    StatevectorAlt<double> svAlt;
    double m[8] = {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2, 0, 0, 0, 0};
    // double m[8] = {1,0,0,1,0,0,0,0};
public:
    TestSepAlt(unsigned nqubits=12)
        : svSep(nqubits, false), svAlt(nqubits, false) {
        name = "test sep and alt";

        addTestCase([&]() -> bool {
            svSep.randomize();
            svAlt.copyValueFrom(svSep);

            u3_f64_sep_0200ffff(svSep.real, svSep.imag, 0, 1 << (svSep.nqubits - 3), m);
            u3_f64_alt_0200ffff(svAlt.data, 0, 1 << (svAlt.nqubits - 2), m);

            return is_close(fidelity(svSep, svAlt), 1);
        }, "test gate H equal (k=2), standard vs sep");
    }
};

int main() {
    auto testSuite = TestSuite();

    auto t0 = TestH { };
    auto t1 = TestSepAlt { };

    testSuite.addTest(&t0);
    testSuite.addTest(&t1);

    testSuite.runAll();

    return 0;
}