#ifndef SIMULATION_MATRICES_H
#define SIMULATION_MATRICES_H

#include "simulation/types.h"

namespace simulation::matrix {

const static ComplexMatrix2 X { {0, 1, 1, 0}, {} };

const static ComplexMatrix4 CZ { {1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,0,-1}, {} };


} // namespace simulation

#endif // SIMULATION_MATRICES_H