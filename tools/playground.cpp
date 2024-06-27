#include "openqasm/parser.h"
// #include "quench/parser.h"
#include "quench/Polynomial.h"
#include "quench/simulate.h"
#include "quench/QuantumGate.h"

#include <random>
#include "utils/iocolor.h"

using namespace Color;
using namespace quench::simulate;
using namespace quench::circuit_graph;
using namespace quench::quantum_gate;

template<typename real_t = double>
class TmpStatevector {
    using complex_t = quench::complex_matrix::Complex<real_t>;
public:
    unsigned nqubits;
    size_t N;
    complex_t* data;
    TmpStatevector(unsigned nqubits) : nqubits(nqubits), N(1<<nqubits) {
        data = new complex_t[N];
    }

    ~TmpStatevector() { delete[] data; }

    TmpStatevector(TmpStatevector&&) = delete;
    TmpStatevector& operator=(TmpStatevector&&) = delete;

    TmpStatevector(const TmpStatevector& that)
        : nqubits(that.nqubits), N(that.N)
    {
        data = new complex_t[N];
        for (size_t i = 0; i < N; i++)
            data[i] = that.data[i];
    }

    TmpStatevector& operator=(const TmpStatevector& that) {
        assert(nqubits == that.nqubits);
        if (this == &that)
            return *this;

        for (size_t i = 0; i < N; i++)
            data[i] = that.data[i];
        return *this;
    }

    double normSquared() const {
        double sum = 0;
        for (size_t i = 0; i < N; i++) {
            sum += data[i].real * data[i].real;
            sum += data[i].imag * data[i].imag;
        }
        return sum;
    }

    double norm() const { return std::sqrt(normSquared()); }

    void normalize() {
        double n = norm();
        for (size_t i = 0; i < N; i++) {
            data[i].real /= n;
            data[i].imag /= n;
        }
    }

    void zeroState() {
        for (size_t i = 0; i < N; i++)
            data[i] = {0.0, 0.0};
        data[0].real = 1.0;
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen { rd() };
        std::normal_distribution<real_t> d { 0, 1 };
        for (size_t i = 0; i < N; i++) {
            data[i].real = d(gen);
            data[i].imag = d(gen);
        }
        normalize();
    }

    std::ostream& print(std::ostream& os) const {
        const auto print_number = [&](size_t idx) {
            if (data[idx].real >= 0)
                os << " ";
            os << data[idx].real;
            if (data[idx].imag >= 0)
                os << " + ";
            os << data[idx].imag << "i";
        };

        if (N > 32) {
            os << BOLD << CYAN_FG << "Warning: " << RESET << "statevector has more "
                "than 5 qubits, only the first 32 entries are shown.\n";
        }
        for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
            os << std::bitset<5>(i) << ": ";
            print_number(i);
            os << "\n";
        }
        return os;
    }
};

int main(int argc, char** argv) {
    openqasm::Parser parser(argv[1], 0);
    auto qasmRoot = parser.parse();
    std::cerr << "qasm AST built\n";
    auto graph = qasmRoot->toCircuitGraph();
    std::cerr << "CircuitGraph built\n";

    // TmpStatevector<double> sv1(graph.nqubits);
    // sv1.zeroState();
    // sv1.randomize();
    // auto sv2 = sv1;
    // auto allBlocks = graph.getAllBlocks();
    // std::vector<GateNode*> gates;
    // std::vector<unsigned> qubits;

    std::cerr << "Before Fusion: " << graph.countBlocks() << " blocks\n";
    // graph.print(std::cerr, 2) << "\n";
    graph.displayInfo(std::cerr, 2) << "\n";
    // for (const auto& block : allBlocks) {
    //     gates.clear();
    //     block->applyInOrder([&gates](GateNode* g) { gates.push_back(g); });

    //     std::cerr << CYAN_FG << BOLD << "gates in block " << block->id << ": ";
    //     for (const auto* gate : gates)
    //         std::cerr << gate->id << ",";
    //     std::cerr << RESET << "\n";

    //     for (const auto& gate : gates) {
    //         qubits.clear();
    //         for (const auto& data : gate->dataVector)
    //             qubits.push_back(data.qubit);

    //         applyGeneral<double>(sv1.data, gate->gateMatrix, qubits, sv1.nqubits);
    //     }
    // }

    for (int m = 2; m < 6; m++) {
        graph.greedyGateFusion(m);
        std::cerr << "After Greedy Fusion " << m << ":\n";
        // graph.print(std::cerr, 2);
        graph.displayInfo(std::cerr, 2) << "\n";
    }
    // allBlocks = graph.getAllBlocks();

    // sv2.print(std::cerr);
    // for (const auto& block : allBlocks) {
    //     gates.clear();
    //     block->applyInOrder([&gates](GateNode* g) { gates.push_back(g); });
    //     std::cerr << CYAN_FG << BOLD << "gates in block " << block->id << ": ";
    //     for (const auto* gate : gates)
    //         std::cerr << gate->id << ",";
    //     std::cerr << RESET << "\n";

    //     for (const auto& gate : gates) {
    //         qubits.clear();
    //         for (const auto& data : gate->dataVector)
    //             qubits.push_back(data.qubit);

    //         applyGeneral<double>(sv2.data, gate->gateMatrix, qubits, sv2.nqubits);
    //     }
    // }
    // sv1.print(std::cerr) << "\n";
    // sv2.print(std::cerr);

    auto m1 = GateMatrix::FromName("u3", {3.1415926, 0.0, 3.1415926});
    m1.matrix.constantMatrix = m1.matrix.constantMatrix.leftKronI();
    m1.nqubits += 1;
    m1.N *= 2;
    m1.printMatrix(std::cerr) << "\n";

    auto m2 = m1.permute({1, 0});
    m2.printMatrix(std::cerr);

    QuantumGate gate1(m1, {4, 3});
    gate1.displayInfo(std::cerr);

    gate1.sortQubits();
    gate1.displayInfo(std::cerr);


    return 0;
}