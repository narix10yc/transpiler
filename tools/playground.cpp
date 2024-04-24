#include "openqasm/parser.h"
#include "qch/ast.h"
#include "simulation/cpu.h"

using namespace simulation;


int main(int argc, char *argv[]) {

    ComplexMatrix4 mat {{
        1.04, 0.22, 0.15, 0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0},
    {0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0}};

    mat.print(std::cerr);

    return 0;
}