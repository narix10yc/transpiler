#include "openqasm/parser.h"
// #include "quench/parser.h"
#include "quench/GateMatrix.h"
#include "quench/simulate.h"

using namespace quench::simulate;
using namespace quench::circuit_graph;

template<typename real_ty = double>
class TmpStatevector {
    using complex_t = quench::cas::Complex<real_ty>;
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
};

int main(int argc, char** argv) {
    openqasm::Parser parser(argv[1], 0);
    auto qasmRoot = parser.parse();
    std::cerr << "qasm AST built\n";
    auto graph = qasmRoot->toCircuitGraph();
    std::cerr << "CircuitGraph built\n";

    TmpStatevector<double> sv1(4);
    auto sv2 = sv1;
    auto allBlocks = graph.getAllBlocks();
    std::vector<GateNode*> gates;
    std::vector<unsigned> qubits;

    std::cerr << "Before Fusion: " << graph.countBlocks() << " blocks\n";
    graph.print(std::cerr, 2) << "\n";
    graph.displayInfo(std::cerr, 2) << "\n";

    for (const auto& block : allBlocks) {
        gates.clear();
        block->applyInOrder([&gates](GateNode* gate) { gates.push_back(gate); });
        std::cerr << "Block " << block->id << " has " << gates.size() << " gates\n";

        for (const auto& gate : gates) {
            qubits.clear();
            for (const auto& data : gate->dataVector)
                qubits.push_back(data.qubit);

            applyGeneral<double>(sv1.data, gate->gateMatrix, qubits, sv1.nqubits);
        }
    }

    for (unsigned maxNqubits = 2; maxNqubits < 3; maxNqubits++) {
        graph.greedyGateFusion(maxNqubits);
        std::cerr << "After Greedy Fusion " << maxNqubits << ":\n";
        graph.print(std::cerr, 2);
        graph.displayInfo(std::cerr, 2) << "\n";
    }

    return 0;
}