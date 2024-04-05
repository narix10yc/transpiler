#ifndef QCH_AST_H_
#define QCH_AST_H_

#include <ostream>
#include <string>
#include <vector>

namespace qch::ast {

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void print(std::ostream&) const = 0;
};


class GateApply : public ASTNode {
    std::string name;
    std::vector<double> parameters;
    std::vector<unsigned> qubits;
public:
    GateApply(std::string name,
              std::initializer_list<double> parameters,
              std::initializer_list<unsigned> qubits)
        : name(name), parameters(parameters), qubits(qubits) {}

    void print(std::ostream&) const override;
};

// class Measure : public ASTNode {
// public:
//     void print(const std::ostream&) const override;
// };

// class IfThenElse : public ASTNode {
// public:
//     void print(const std::ostream&) const override;
// };

} // namespace qch

#endif // QCH_AST_H_