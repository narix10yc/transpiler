#ifndef QCH_AST_H_
#define QCH_AST_H_

#include <string>
#include <set>
#include <unordered_set>
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>

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

    Statement* getStmtPtr(size_t index) const {
        return stmts[index].get();
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
public:
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

    size_t countStmts() const { return stmts.size(); }

    Statement* getStmtPtr(size_t index) const { return stmts[index].get(); }

    unsigned getNumQubits() const { return nqubits; }

    void setNumQubits(unsigned n) { nqubits = n; }

    void print(std::ostream&) const override;

    void genCPU(simulation::CPUGenContext& ctx) const override;
};

// forward declaration
class CASContext;

class CASNode {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual ~CASNode() = default;
    
    virtual expr_value getExprValue() const = 0;

    virtual CASNode* canonicalize(CASContext& ctx) const = 0;

    virtual CASNode* derivative(const std::string& var,
                                CASContext& ctx) const = 0;
    
    virtual void print(std::ostream& os) const = 0;

    // virtual void printLatex(std::ostream& os) const = 0;
};


class CASContext {
    std::vector<CASNode*> nodes;
public: 
    CASContext() : nodes() {}

    CASContext(CASContext&) = delete;
    CASContext(CASContext&&) = delete;
    CASContext& operator=(CASContext&) = delete;
    CASContext& operator=(CASContext&&) = delete;
    ~CASContext() {
        for (auto node : nodes)
            delete(node);
    }

public:
    CASNode* addNode(CASNode* node) {
        nodes.push_back(node);
        return node;
    }

    CASNode* getConstant(double value);

    CASNode* getVariable(const std::string& name);

    /// @brief Add
    CASNode* getAdd(CASNode* lhs, CASNode* rhs);
    CASNode* getAdd(CASNode* lhs, double value) {
        return getAdd(lhs, getConstant(value));
    }
    CASNode* getAdd(CASNode* lhs, const std::string& var) {
        return getAdd(lhs, getVariable(var));
    }

    /// @brief Sub 
    CASNode* getSub(CASNode* lhs, CASNode* rhs);
    CASNode* getSub(CASNode* lhs, double value) {
        return getSub(lhs, getConstant(value));
    }
    CASNode* getSub(CASNode* lhs, const std::string& var) {
        return getSub(lhs, getVariable(var));
    }

    /// @brief Mul
    CASNode* getMul(CASNode* lhs, CASNode* rhs);
    CASNode* getMul(CASNode* lhs, double value) {
        return getMul(getConstant(value), lhs);
    }
    CASNode* getMul(CASNode* lhs, const std::string& var) {
        return getMul(getVariable(var), lhs);
    }

    /// @brief Pow
    CASNode* getPow(CASNode* base, CASNode* exponent);
    CASNode* getPow(CASNode* lhs, double value) {
        return getPow(lhs, getConstant(value));
    }

    /// @brief Neg
    CASNode* getNeg(CASNode* node);

    /// @brief Cos
    CASNode* getCos(CASNode* node);

    /// @brief Sin
    CASNode* getSin(CASNode* node);
};


class CASGraphExpr : public Node {
    CASContext ctx;
    CASNode* entry;
public:
    CASGraphExpr() : ctx(), entry(nullptr) {}

    CASContext& getContext() { return ctx; }

    CASNode* getConstant(double value) {
        return ctx.getConstant(value);
    }
    CASNode* getVariable(const std::string& name) {
        return ctx.getVariable(name);
    }

    /// @brief Add
    CASNode* getAdd(CASNode* lhs, CASNode* rhs) {
        return entry = ctx.getAdd(lhs, rhs);
    }
    CASNode* getAdd(CASNode* lhs, double value) {
        return getAdd(lhs, getConstant(value));
    }
    CASNode* getAdd(CASNode* lhs, const std::string& var) {
        return getAdd(lhs, getVariable(var));
    }

    /// @brief Sub
    CASNode* getSub(CASNode* lhs, CASNode* rhs) {
        return entry = ctx.getSub(lhs, rhs);
    }
    CASNode* getSub(CASNode* lhs, double value) {
        return getSub(lhs, getConstant(value));
    }
    CASNode* getSub(CASNode* lhs, const std::string& var) {
        return getSub(lhs, getVariable(var));
    }

    /// @brief Mul
    CASNode* getMul(CASNode* lhs, CASNode* rhs) {
        return entry = ctx.getMul(lhs, rhs);
    }
    CASNode* getMul(CASNode* lhs, double value) {
        return getMul(getConstant(value), lhs);
    }
    CASNode* getMul(CASNode* lhs, const std::string& var) {
        return getMul(getVariable(var), lhs);
    }

    /// @brief Pow
    CASNode* getPow(CASNode* base, CASNode* exponent) {
        return entry = ctx.getPow(base, exponent);
    }
    CASNode* getPow(CASNode* lhs, double value) {
        return getPow(lhs, getConstant(value));
    }

    /// @brief Neg
    CASNode* getNeg(CASNode* node) {
        return entry = ctx.getNeg(node);
    }

    /// @brief Cos
    CASNode* getCos(CASNode* node) {
        return entry = ctx.getCos(node);
    }

    /// @brief Sin
    CASNode* getSin(CASNode* node) {
        return entry = ctx.getSin(node);
    }

    void setEntry(CASNode* node) { entry = node; }

    void print(std::ostream& os) const override {
        assert(entry != nullptr);
        entry->print(os);
    }

    void genCPU(simulation::CPUGenContext&) const override {}

    CASNode* computeCanonicalize(CASNode* node) {
        return node->canonicalize(ctx);
    }
    CASNode* computeCanonicalize() {
        return computeCanonicalize(entry);
    }

    CASNode* computeDerivative(const std::string& var, CASNode* node) {
        return node->derivative(var, ctx);
    }

    CASNode* computeDerivative(const std::string& var) {
        return computeDerivative(var, entry);
    }

};


class CASConstant : public CASNode {
    double value;
public:
    CASConstant(double value) : value(value) {}

    double getValue() const { return value; }

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;

};

class CASVariable : public CASNode {
    std::string name;
public:
    CASVariable(std::string name) : name(name) {}

    std::string getName() const { return name; }

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;

};

class CASAdd : public CASNode {
    CASNode* lhs;
    CASNode* rhs;
public:
    CASAdd(CASNode* lhs, CASNode* rhs) : lhs(lhs), rhs(rhs) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASSub : public CASNode {
    CASNode* lhs;
    CASNode* rhs;
public:
    CASSub(CASNode* lhs, CASNode* rhs) : lhs(lhs), rhs(rhs) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASMul : public CASNode {
    CASNode* lhs;
    CASNode* rhs;
public:
    CASMul(CASNode* lhs, CASNode* rhs) : lhs(lhs), rhs(rhs) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASPow : public CASNode {
    CASNode* base;
    CASNode* exponent;
public:
    CASPow(CASNode* base, CASNode* exponent) : base(base), exponent(exponent) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASNeg : public CASNode {
    CASNode* node;
public:
    CASNeg(CASNode* node) : node(node) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASCos : public CASNode {
    CASNode* node;
public:
    CASCos(CASNode* node) : node(node) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};

class CASSin : public CASNode {
    CASNode* node;
public:
    CASSin(CASNode* node) : node(node) {}

    expr_value getExprValue() const override;

    CASNode* canonicalize(CASContext& ctx) const override;

    CASNode* derivative(const std::string& var,
                        CASContext& ctx) const override;

    void print(std::ostream& os) const override;
};





} // namespace qch

#endif // QCH_AST_H_w