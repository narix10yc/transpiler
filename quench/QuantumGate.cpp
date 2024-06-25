#include "quench/QuantumGate.h"
#include "utils/iocolor.h"

using namespace Color;
using namespace quench::complex_matrix;
using namespace quench::quantum_gate;

GateMatrix::GateMatrix(std::initializer_list<Complex<double>> m) {
    std::cerr << "Constructing ConstantMatrix\n";
    auto mSize = m.size();
    matrix.activeType = matrix_t::ActiveMatrixType::C;

    new (&matrix.constantMatrix) matrix_t::c_matrix_t(m);

    // gateMatrix.matrix.constantMatrix = matrix_t::c_matrix_t(m);
    std::cerr << "Hello!\n";
    switch (mSize) {
    case 0:
        assert(false && "Empty matrix");
        break;
    case 1:
        assert(false && "1x1 matrix does not represent quantum gates");
        break;
    case 4: nqubits = 1; break;
    case 16: nqubits = 2; break;
    case 64: nqubits = 3; break;
    case 256: nqubits = 4; break;
    default:
        nqubits = std::log2(mSize);
        assert(std::pow(2, nqubits) == mSize);
        break;
    }
    N = 1 << nqubits;
}

GateMatrix GateMatrix::FromName(const std::string& name,
                                const std::vector<double>& params)
{
    if (name == "u3") {
        assert(params.size() == 3);
        const double theta = 0.5 * params[0];
        const auto& phi = params[1];
        const auto& lambd = params[2];
        const double ctheta = std::cos(theta);
        const double stheta = std::sin(theta);

        return { 
            {ctheta, 0},
            {-std::cos(lambd) * stheta, -std::sin(lambd) * stheta},
            {std::cos(phi) * stheta, std::sin(phi) * stheta},
            {std::cos(phi + lambd) * ctheta, std::sin(phi + lambd) * ctheta}
        };
    }
    if (name == "cx") {
        return {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {1,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0}
        };
    }

    if (name == "cz") {
        return {
            {1,0}, {0,0}, {0,0}, {0,0},
            {0,0}, {1,0}, {0,0}, {0,0},
            {0,0}, {0,0}, {1,0}, {0,0},
            {0,0}, {0,0}, {0,0}, {-1,0}
        };
    }

    std::cerr << RED_FG << BOLD << "Error: Unrecognized gate '" << name << "'"
              << RESET << "\n";
    return GateMatrix();
}