#include "unit_test.h"
#include "simulation/tplt.h"
#include "simulation/statevector.h"

#include <cmath>

#define INV_SQRT2 (0.7071067811865475727373109)

extern "C" {
void f64_s2_alt_u3_k3_33330333(double*, uint64_t, uint64_t, void*);

void f64_s2_sep_u3_k3_33330333(double*, double*, uint64_t, uint64_t, void*);
void f64_s2_sep_u3_k0_33330333(double*, double*, uint64_t, uint64_t, void*);
void f64_s2_sep_u3_k1_33330333(double*, double*, uint64_t, uint64_t, void*);

void f64_s1_sep_u2q_k2l1_ffffffffffffffff(double*, double*, uint64_t, uint64_t, void*);
void f64_s2_sep_u2q_k5l3_ffffffffffffffff(double*, double*, uint64_t, uint64_t, void*);
void f64_s2_sep_u2q_k1l0_ffffffffffffffff(double*, double*, uint64_t, uint64_t, void*);
void f64_s1_sep_u2q_k2l0_ffffffffffffffff(double*, double*, uint64_t, uint64_t, void*);

}

using namespace simulation;
using namespace simulation::sv;
using namespace simulation::tplt;
using namespace simulation::test;


bool is_close(double a, double b, double thres=1e-8) {
    return abs(a - b) < thres;
}

class TestH : public Test {
    unsigned n; // nqubits
    StatevectorSep<double> svSep1, svSep2;
    ComplexMatrix2<double> mat {{INV_SQRT2,INV_SQRT2,INV_SQRT2,-INV_SQRT2}, {0,0,0,0}};
    double m[8] = {INV_SQRT2, INV_SQRT2, INV_SQRT2, -INV_SQRT2, 0, 0, 0, 0};
public:
    TestH(unsigned _nqubits=12)
        : n(_nqubits), svSep1(_nqubits), svSep2(_nqubits) {
        name = "test H";

        addTestCase([&]() -> bool {
            svSep1.randomize();

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, 0);
            return is_close(svSep1.normSquared(), 1.0);
        }, "test gate H norm (k=0), standard method");

        addTestCase([&]() -> bool {
            svSep1.randomize();

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, n/2);
            return is_close(svSep1.normSquared(), 1.0);
        }, "test gate H norm (k = n/2), standard method");

        addTestCase([&]() -> bool {
            svSep1.randomize();
            svSep2 = svSep1;

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, 0);
            f64_s2_sep_u3_k0_33330333(svSep2.real, svSep2.imag, 0, 1<<(n-3), m);
            return is_close(fidelity(svSep1, svSep2), 1.0);
        }, "test gate H (k=0), standard vs sep");
    }
};

class TestU3 : public Test {
    unsigned n; // nqubits
    StatevectorSep<double> svSep1, svSep2;
    StatevectorAlt<double> svAlt;
public:
    TestU3(unsigned _nqubits=12)
        : n(_nqubits), svSep1(_nqubits), svSep2(_nqubits), svAlt(_nqubits) {
        name = "test U3";

        addTestCase([&]() -> bool {
            svSep1.randomize();
            svSep2 = svSep1;

            auto mat = ComplexMatrix2<double>::Random();
            double m[8];
            for (size_t i = 0; i < 4; i++) {
                m[i] = mat.real[i];
                m[i+4] = mat.imag[i];
            }

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, 0);
            f64_s2_sep_u3_k0_33330333(svSep2.real, svSep2.imag, 0, 1<<(n-3), m);

            svSep1.normalize(); svSep2.normalize();
            return is_close(fidelity(svSep1, svSep2), 1.0);
        }, "test gate U3 k=0, standard vs sep");

        addTestCase([&]() -> bool {
            svSep1.randomize();
            svSep2 = svSep1;

            auto mat = ComplexMatrix2<double>::Random();
            double m[8];
            for (size_t i = 0; i < 4; i++) {
                m[i] = mat.real[i];
                m[i+4] = mat.imag[i];
            }

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, 1);
            f64_s2_sep_u3_k1_33330333(svSep2.real, svSep2.imag, 0, 1<<(n-3), m);

            svSep1.normalize(); svSep2.normalize(); svAlt.normalize();
            return is_close(fidelity(svSep1, svSep2), 1.0);
        }, "test gate U3 k=1, standard vs sep");

        addTestCase([&]() -> bool {
            svSep1.randomize();
            svSep2 = svSep1;
            svAlt.copyValueFrom(svSep1);

            auto mat = ComplexMatrix2<double>::Random();
            double m[8];
            for (size_t i = 0; i < 4; i++) {
                m[i] = mat.real[i];
                m[i+4] = mat.imag[i];
            }

            applySingleQubit<double>(svSep1.real, svSep1.imag, mat, n, 3);
            f64_s2_sep_u3_k3_33330333(svSep2.real, svSep2.imag, 0, 1<<(n-3), m);
            f64_s2_alt_u3_k3_33330333(svAlt.data, 0, 1<<(n-2), m);

            svSep1.normalize(); svSep2.normalize(); svAlt.normalize();
            return is_close(fidelity(svSep1, svSep2), 1.0) && is_close(fidelity(svSep1, svAlt), 1.0);
        }, "test gate U3 k=3, standard vs sep vs alt");
    }
};

class TestU2q : public Test {
    unsigned n; // nqubits
    StatevectorSep<double> sv0, sv1;
public:
    TestU2q(unsigned _nqubits=12)
        : n(_nqubits), sv0(_nqubits), sv1(_nqubits) {
        name = "u2q gate sep format";

        // addTestCase([&]() -> bool {
        //     sv0.randomize();
        //     sv1 = sv0;

        //     U2qGate u2q { 0, 1, ComplexMatrix4<>::Random() };
        //     applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, n, u2q.k, u2q.l);
        //     u2q.swapTargetQubits();
        //     applyTwoQubitQuEST<double>(sv1.real, sv1.imag, u2q.mat, n, u2q.k, u2q.l);

        //     sv0.normalize(); sv1.normalize();
        //     return is_close(fidelity(sv0, sv1), 1.0);
        // }, "swap qubits");

        // addTestCase([&]() -> bool {
        //     sv0.randomize();
        //     sv1 = sv0;

        //     auto mat = ComplexMatrix4<>::Random();
        //     double m[32];
        //     for (size_t i = 0; i < 16; i++) {
        //         m[i] = mat.real[i];
        //         m[i+16] = mat.imag[i];
        //     }
            
        //     applyTwoQubitQuEST<double>(sv0.real, sv0.imag, mat, n, 2, 1);
        //     f64_s1_sep_u2q_k2l1_ffffffffffffffff(sv1.real, sv1.imag, 0, 1<<(n-3), m);

        //     sv0.normalize(); sv1.normalize();
        //     return is_close(fidelity(sv0, sv1), 1.0);
        // }, "quest and ir kernel result match, s=1, k=2, l=1");

        // addTestCase([&]() -> bool {
        //     sv0.randomize();
        //     sv1 = sv0;

        //     U2qGate u2q { 5, 3, ComplexMatrix4<>::Random() };
        //     double m[32];
        //     for (size_t i = 0; i < 16; i++) {
        //         m[i] = u2q.mat.real[i];
        //         m[i+16] = u2q.mat.imag[i];
        //     }
            
        //     applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, n, u2q.k, u2q.l);
        //     f64_s2_sep_u2q_k5l3_ffffffffffffffff(sv1.real, sv1.imag, 0, 1<<(n-4), m);

        //     sv0.normalize(); sv1.normalize();
        //     return is_close(fidelity(sv0, sv1), 1.0);
        // }, "quest and ir kernel result match, s=2, k=5, l=3");

        // addTestCase([&]() -> bool {
        //     sv0.randomize();
        //     sv1 = sv0;

        //     U2qGate u2q { 1, 0, ComplexMatrix4<>::Random() };
        //     double m[32];
        //     for (size_t i = 0; i < 16; i++) {
        //         m[i] = u2q.mat.real[i];
        //         m[i+16] = u2q.mat.imag[i];
        //     }
            
        //     applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, n, u2q.k, u2q.l);
        //     f64_s2_sep_u2q_k1l0_ffffffffffffffff(sv1.real, sv1.imag, 0, 1<<(n-4), m);

        //     sv0.normalize(); sv1.normalize();
        //     return is_close(fidelity(sv0, sv1), 1.0);
        // }, "quest and ir kernel result match, s=2, k=1, l=0 (shuffle needed)");

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            auto mat = ComplexMatrix4<>::Random();
            // ComplexMatrix4<> mat {
                // {1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0}, {}
            // };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = mat.real[i];
                m[i+16] = mat.imag[i];
            }
            
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, mat, n, 2, 0);
            f64_s1_sep_u2q_k2l0_ffffffffffffffff(sv1.real, sv1.imag, 0, 1<<(n-3), m);

            sv0.print(std::cerr);
            std::cerr << "\n";
            sv1.print(std::cerr);

            sv0.normalize(); sv1.normalize();
            return is_close(fidelity(sv0, sv1), 1.0);
        }, "quest and ir kernel result match, s=1, k=2, l=0 (shuffle needed)");
    }
};


class TestU2qBatched : public Test {
    unsigned n; // nqubits
    StatevectorSep<double> sv0, sv1, sv2;
public:
    TestU2qBatched(unsigned _nqubits=12)
        : n(_nqubits), sv0(_nqubits), sv1(_nqubits), sv2(_nqubits) {
        name = "u2q gate batched";

        addTestCase([&]() -> bool {
            sv0.randomize();
            sv1 = sv0;

            U2qGate u2q { 2, 1, ComplexMatrix4<>::Random() };
            double m[32];
            for (size_t i = 0; i < 16; i++) {
                m[i] = u2q.mat.real[i];
                m[i+16] = u2q.mat.imag[i];
            }
            applyTwoQubitQuEST<double>(sv0.real, sv0.imag, u2q.mat, n, u2q.k, u2q.l);

            // f64_s1_sep_u2q_k2l1_batched(sv1.real, sv1.imag, 
                        // sv2.real, sv2.imag, 0, 1 << (n - 3), m);

            sv0.normalize();
            sv2.normalize();

            return is_close(fidelity(sv0, sv2), 1, 1e-8);
        }, "swap qubits");
    
    }
};

int main() {
    auto testSuite = TestSuite();

    auto t0 = TestH { };
    auto t1 = TestU3 { };
    auto t2 = TestU2q { };

    testSuite.addTest(&t0);
    testSuite.addTest(&t1);
    testSuite.addTest(&t2);

    testSuite.runAll();

    return 0;
}