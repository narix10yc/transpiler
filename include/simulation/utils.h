#ifndef SIMULATION_UTILS_H_
#define SIMULATION_UTILS_H_

#include "simulation/types.h"

namespace simulation {

/// @return a random angle between -pi and pi 
double randomAngle();

/// @return a random 2x2 complex matrix. Each entry is Gaussian distributed
ComplexMatrix2<double> randomComplexMatrix2();

/// @return a random 4x4 complex matrix. Each entry is Gaussian distributed
ComplexMatrix4<double> randomComplexMatrix4();


} // namespace simulation

#endif // SIMULATION_UTILS_H_
