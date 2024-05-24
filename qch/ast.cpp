#include "qch/ast.h"
#include <cmath>

using namespace qch::ast;
using expr_value = qch::ast::CASExpr::expr_value;

void GateApplyStmt::print(std::ostream& os) const {
    os << "  " << name;
    auto pSize = parameters.size();

    // parameter
    if (pSize > 0) {
        os << "(";
        for (size_t i = 0; i < pSize-1; i++)
            os << parameters[i] << ",";
        os << parameters[pSize-1] << ")";
    }

    // qubits
    os << " ";
    auto qSize = qubits.size();
    for (size_t i = 0; i < qSize-1; i++)
        os << qubits[i] << " ";

    os << qubits[qSize-1] << "\n";
}

void CircuitStmt::print(std::ostream& os) const {
    os << "circuit<" << nqubits << "> " << name << "()\n{\n";
    
    for (auto& s : stmts)
        s->print(os);

    os << "}\n";
}


void CASConstant::print(std::ostream& os) const {
    os << value;
}

void CASVariable::print(std::ostream& os) const {
    os << name;
}

void CASAdd::print(std::ostream& os) const {
    lhs->print(os);
    os << " + ";
    rhs->print(os);
}

void CASSub::print(std::ostream& os) const {
    lhs->print(os);
    os << " - ";
    rhs->print(os);
}

void CASMul::print(std::ostream& os) const {
    lhs->print(os);
    os << " * ";
    rhs->print(os);
}

void CASNeg::print(std::ostream& os) const {
    os << "-";
    expr->print(os);
}

void CASCos::print(std::ostream& os) const {
    os << "cos(";
    expr->print(os);
    os << ")";
}

void CASSin::print(std::ostream& os) const {
    os << "sin(";
    expr->print(os);
    os << ")";
}

expr_value CASConstant::getExprValue() const {
    return { true, value };
}

expr_value CASVariable::getExprValue() const {
    return { false };
}

expr_value CASAdd::getExprValue() const {
    auto vL = lhs->getExprValue();
    auto vR = rhs->getExprValue();
    if (vL.isConstant && vR.isConstant)
        return { true, vL.value + vR.value };
    return { false };
}

expr_value CASSub::getExprValue() const {
    auto vL = lhs->getExprValue();
    auto vR = rhs->getExprValue();
    if (vL.isConstant && vR.isConstant)
        return { true, vL.value - vR.value };
    return { false };
}

expr_value CASMul::getExprValue() const {
    auto vL = lhs->getExprValue();
    auto vR = rhs->getExprValue();
    if (vL.isConstant && vR.isConstant)
        return { true, vL.value * vR.value };
    return { false };
}

expr_value CASNeg::getExprValue() const {
    auto v = expr->getExprValue();
    if (v.isConstant)
        return { true, -v.value };
    return { false };
}

expr_value CASCos::getExprValue() const {
    auto v = expr->getExprValue();
    if (v.isConstant)
        return { true, std::cos(v.value) };
    return { false };
}

expr_value CASSin::getExprValue() const {
    auto v = expr->getExprValue();
    if (v.isConstant)
        return { true, std::sin(v.value) };
    return { false };
}


std::unique_ptr<CASExpr> CASConstant::canonicalize() const {
    return std::make_unique<CASConstant>(this);
}

std::unique_ptr<CASExpr> CASVariable::canonicalize() const {
    return std::make_unique<CASVariable>(this);
}

std::unique_ptr<CASExpr> CASAdd::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASAdd>(this);
}

std::unique_ptr<CASExpr> CASSub::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASSub>(this);
}

std::unique_ptr<CASExpr> CASMul::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASMul>(this);
}

std::unique_ptr<CASExpr> CASNeg::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASNeg>(this);
}

std::unique_ptr<CASExpr> CASCos::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASCos>(this);
}

std::unique_ptr<CASExpr> CASSin::canonicalize() const {
    auto v = getExprValue();
    if (v.isConstant)
        return std::make_unique<CASConstant>(v.value);
    return std::make_unique<CASSin>(this);
}


std::unique_ptr<CASExpr> CASConstant::derivative(const std::string& var) const {
    return std::make_unique<CASConstant>(0.0);
}

std::unique_ptr<CASExpr> CASVariable::derivative(const std::string& var) const {
    return std::make_unique<CASConstant>((var == name) ? 1.0 : 0.0);
}

std::unique_ptr<CASExpr> CASAdd::derivative(const std::string& var) const {
    return std::make_unique<CASAdd>(lhs->derivative(var), rhs->derivative(var));
}

std::unique_ptr<CASExpr> CASSub::derivative(const std::string& var) const {
    return std::make_unique<CASSub>(lhs->derivative(var), rhs->derivative(var));
}

std::unique_ptr<CASExpr> CASMul::derivative(const std::string& var) const {
    auto t1 = std::make_unique<CASMul>(std::move(lhs), rhs->derivative(var));
    auto t2 = std::make_unique<CASMul>(lhs->derivative(var), std::move(rhs));

    return std::make_unique<CASAdd>(std::move(t1), std::move(t2));
}

std::unique_ptr<CASExpr> CASCos::derivative(const std::string& var) const {
    // cos(f(x))' = -sin(f(x)) * f'(x)
    auto t1 = std::make_unique<CASSin>(std::move(expr)); //  sin(f(x))
    auto t2 = std::make_unique<CASSub>(std::move(t1));   // -sin(f(x))
    return std::make_unique<CASMul>(std::move(t2), expr->derivative(var));
}

std::unique_ptr<CASExpr> CASSin::derivative(const std::string& var) const {
    // sin(f(x))' = cos(f(x)) * f'(x)
    auto t1 = std::make_unique<CASCos>(std::move(expr)); // cos(f(x))
    return std::make_unique<CASMul>(std::move(t1), expr->derivative(var));
}