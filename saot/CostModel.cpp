#include "saot/CostModel.h"
#include "saot/QuantumGate.h"

using namespace saot;

int StandardCostModel::getCost(const QuantumGate& gate) const {
    assert(0 && "Not Implemented");
    return -1;
}

int AdaptiveCostModel::getCost(const QuantumGate& gate) const {
    assert(0 && "Not Implemented");
    return -1;
}