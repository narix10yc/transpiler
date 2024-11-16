#ifndef SAOT_COSTMODEL_H
#define SAOT_COSTMODEL_H

#include <cassert>

namespace saot {

class QuantumGate;

class CostModel {
public:
    ~CostModel() = default;

    virtual int getCost(const QuantumGate& gate) const {
        assert(false && "Should not call from base class");
        return -1;
    }

};

class StandardCostModel : public CostModel {
    int maxNQubits;
    int maxOp;
public:
    StandardCostModel(int maxNQubits, int maxOp)
        : maxNQubits(maxNQubits), maxOp(maxOp) {}
        
    int getCost(const QuantumGate& gate) const override;
};

class AdaptiveCostModel : public CostModel {
public:
    int getCost(const QuantumGate& gate) const override;

};

} // namespace saot;

#endif // SAOT_COSTMODEL_H