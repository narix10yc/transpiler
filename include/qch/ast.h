#ifndef QCH_AST_H_
#define QCH_AST_H_

#include <string>
#include <vector>
#include <iostream>

namespace qch::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual void print(std::ostream&) const = 0;
};

class Statement : public Node {
public:
    Statement() {}
    void print(std::ostream&) const override {}
};

class RootNode : public Node {
    std::vector<std::unique_ptr<Statement>> stmts;
public:
    void addStmt(std::unique_ptr<Statement> stmt) {
        stmts.push_back(std::move(stmt));
    }

    size_t countStmt() const {
        return stmts.size();
    }

    const Statement& getStmt(size_t index) const {
        return *stmts[index];
    }

    void print(std::ostream& os) const override {
        for (auto& s : stmts) {
            s->print(os);
        }
    }
};


class GateApplyStmt : public Statement {
    std::string name;
    std::vector<double> parameters;
    std::vector<unsigned> qubits;
public:
    GateApplyStmt(std::string name) : name(name), parameters(), qubits() {}

    GateApplyStmt(std::string name,
                  std::initializer_list<double> parameters,
                  std::initializer_list<unsigned> qubits)
        : name(name), parameters(parameters), qubits(qubits) {}
    
    void addParameter(double p) { parameters.push_back(p); }
    void addTargetQubit(unsigned q) { qubits.push_back(q); }

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