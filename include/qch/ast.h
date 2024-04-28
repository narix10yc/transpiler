#ifndef QCH_AST_H_
#define QCH_AST_H_

#include <string>
#include <memory>
#include <vector>
#include <iostream>

namespace simulation {
class CPUGenContext;
}


namespace qch::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual void print(std::ostream&) const = 0;
    virtual void genCPU(simulation::CPUGenContext&) const = 0;
};

class Statement : public Node {
public:
    Statement() {}
    void print(std::ostream&) const override {}
    virtual unsigned getLargestQubit() const { return 0; }
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
        std::cerr << "RootNode print\n";
        for (auto& s : stmts) {
            s->print(os);
        }
    }

    void genCPU(simulation::CPUGenContext& ctx) const override {
        for (auto& s : stmts)
            s->genCPU(ctx);
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

    unsigned getLargestQubit() const override {
        unsigned m = 0;
        for (auto q : qubits) {
            if (q > m) m = q;
        }
        return m;
    }

    void print(std::ostream&) const override;

    void genCPU(simulation::CPUGenContext& ctx) const override;
};


class CircuitStmt : public Statement {
    std::string name;
    unsigned nqubits;
    // TODO: circuit parameters
    std::vector<std::unique_ptr<Statement>> stmts;
public:
    CircuitStmt(std::string name, unsigned nqubits=0)
        : name(name), nqubits(nqubits), stmts() {}

    unsigned getLargestQubit() const override { return nqubits - 1; }

    void addStmt(std::unique_ptr<Statement> stmt) {
        // update number of qubits
        unsigned m = stmt->getLargestQubit();

        if (nqubits <= m) {
            nqubits = m + 1;
        }
        stmts.push_back(std::move(stmt));
    }

    unsigned getNumQubits() const { return nqubits; }
    void setNumQubits(unsigned n) { nqubits = n; }

    void print(std::ostream&) const override;

    void genCPU(simulation::CPUGenContext& ctx) const override;
};



} // namespace qch

#endif // QCH_AST_H_