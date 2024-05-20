#include "unit_test.h"
#include "simulation/tplt.h"
#include "simulation/statevector.h"

#include <cmath>

#define INV_SQRT2 (0.7071067811865475727373109)

extern "C" {
    void u3_f64_alt_0200ffff(double*, uint64_t, uint64_t, void*);
    void u3_f64_alt_0000ffff(double*, uint64_t, uint64_t, void*);
    void u3_f64_sep_0200ffff(double*, double*, uint64_t, uint64_t, void*);
    void f64_sep_u3_k0_33330333(double*, double*, uint64_t, uint64_t, void*);
    void f64_sep_u3_k1_33330333(double*, double*, uint64_t, uint64_t, void*);
    void f32_s2_sep_u3_k0_33330333(float*, float*, uint64_t, uint64_t, void*);
    void f32_s2_sep_u3_k1_33330333(float*, float*, uint64_t, uint64_t, void*);
    void f64_s2_sep_u2q_k5l3(double*, double*, uint64_t, uint64_t, void*);
    void f64_s1_sep_u2q_k2l1(double*, double*, uint64_t, uint64_t, void*);
    void f64_s2_sep_u2q_k1l0(double*, double*, uint64_t, uint64_t, void*);
    void f64_s1_sep_u2q_k2l0(double*, double*, uint64_t, uint64_t, void*);
    void f64_s1_sep_u2q_k2l1_batched(double*, double*, double*, double*, uint64_t, uint64_t, void*);
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

class TestSepShuffle : public Test {
    StatevectorSep<double> sv0, sv1;
    double m[8] = {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2, 0, 0, 0, 0};
public:
    TestSepShuffle(unsigned nqubits=12)
        : sv0(nqubits, false), sv1(nqubits, false) {
        name = "test sep shuffled vectorization";

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;
            applySingleQubit<double>(sv0.real, sv0.imag, {
                {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2}, {0, 0, 0, 0}},
                sv0.nqubits, 0);
            f64_sep_u3_k0_33330333(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits - 3), m);
            return is_close(fidelity(sv0, sv1), 1);
        });

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;
            applySingleQubit<double>(sv0.real, sv0.imag, {
                {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2}, {0, 0, 0, 0}},
                sv0.nqubits, 1);
            f64_sep_u3_k1_33330333(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits - 3), m);
            return is_close(fidelity(sv0, sv1), 1);
        });
    }
};

class TestSepShuffleF32 : public Test {
    StatevectorSep<float> sv0, sv1;
    float m[8] = {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2, 0, 0, 0, 0};
public:
    TestSepShuffleF32(unsigned nqubits=12)
        : sv0(nqubits, false), sv1(nqubits, false) {
        name = "test sep shuffled vectorization";

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;
            applySingleQubit<float>(sv0.real, sv0.imag, {
                {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2}, {0, 0, 0, 0}},
                sv0.nqubits, 0);
            f32_s2_sep_u3_k0_33330333(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits - 3), m);
            return is_close(fidelity(sv0, sv1), 1, 1e-5);
        });

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;
            applySingleQubit<float>(sv0.real, sv0.imag, {
                {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2}, {0, 0, 0, 0}},
                sv0.nqubits, 1);
            f32_s2_sep_u3_k1_33330333(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits - 3), m);
            return is_close(fidelity(sv0, sv1), 1, 1e-5);
        });
    }
};

class TestU2qSep : public Test {
    StatevectorSep<double> sv0, sv1;
public:
    TestU2qSep(unsigned nqubits=12)
        : sv0(nqubits), sv1(nqubits) {
        name = "u2q gate sep format";

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 0, 1, ComplexMatrix4<>::Random() };
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);
            u2q.swapTargetQubits();
            applyTwoQubitQuEST<double>(sv1.real, sv1.imag, u2q.mat, sv1.nqubits, u2q.k, u2q.l);

            sv0.normalize();
            sv1.normalize();

            return is_close(fidelity(sv0, sv1), 1, 1e-8);
        }, "swap qubits");

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 2, 1, ComplexMatrix4<>::Random() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);
            f64_s1_sep_u2q_k2l1(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits-3), m);

            sv0.normalize();
            sv1.normalize();

            return is_close(fidelity(sv0, sv1), 1, 1e-8);
        }, "quest and ir kernel result match, s=1, k=2, l=1");

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 5, 3, ComplexMatrix4<>::Random() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);
            f64_s2_sep_u2q_k5l3(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits-4), m);

            sv0.normalize();
            sv1.normalize();

            return is_close(fidelity(sv0, sv1), 1, 1e-8);
        }, "quest and ir kernel result match, s=2, k=5, l=3");

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 1, 0, ComplexMatrix4<>::Random() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);
            f64_s2_sep_u2q_k1l0(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits-4), m);

            sv0.normalize();
            sv1.normalize();

            return is_close(fidelity(sv0, sv1), 1, 1e-8);
        }, "quest and ir kernel result match, s=2, k=1, l=0 (shuffle needed)");

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 2, 0, ComplexMatrix4<>::Identity() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);
            f64_s1_sep_u2q_k2l0(sv1.real, sv1.imag, 0, 1 << (sv1.nqubits-3), m);

            sv0.normalize();
            sv1.normalize();
            
            return is_close(fidelity(sv0, sv1), 1, 1e-8);
        }, "quest and ir kernel result match, s=1, k=2, l=0 (shuffle needed)");
    }
};


class TestU2qSepBatched : public Test {
    StatevectorSep<double> sv0, sv1, sv2;
public:
    TestU2qSepBatched(unsigned nqubits=12)
        : sv0(nqubits), sv1(nqubits), sv2(nqubits) {
        name = "u2q gate sep format";

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 2, 1, ComplexMatrix4<>::Random() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, sv0.nqubits, u2q.k, u2q.l);

            f64_s1_sep_u2q_k2l1_batched(sv1.real, sv1.imag, 
                        sv2.real, sv2.imag, 0, 1 << (sv1.nqubits - 3), m);

            sv0.normalize();
            sv2.normalize();

            return is_close(fidelity(sv0, sv2), 1, 1e-8);
        }, "swap qubits");
    
    }
};

int main() {
    auto testSuite = TestSuite();

    auto t0 = TestH { };
    auto t1 = TestSepAlt { };
    auto t2 = TestSepShuffle { };
    auto t3 = TestSepShuffleF32 { 4 };
    auto t4 = TestU2qSep { };
    auto t5 = TestU2qSepBatched { };

    testSuite.addTest(&t0);
    testSuite.addTest(&t1);
    testSuite.addTest(&t2);
    testSuite.addTest(&t3);
    testSuite.addTest(&t4);
    testSuite.addTest(&t5);

    testSuite.runAll();

    return 0;
}