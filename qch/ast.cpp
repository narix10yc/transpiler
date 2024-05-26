#include "qch/ast.h"
#include <cmath>

using namespace qch::ast;
using expr_value = qch::ast::CASNode::expr_value;

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

/* #region Hello */
CASNode* CASContext::getConstant(double value) {
    return addNode(new CASConstant(value));
}

CASNode* CASContext::getVariable(const std::string& name) {
    return addNode(new CASVariable(name));
}

CASNode* CASContext::getAdd(CASNode* lhs, CASNode* rhs) {
    return addNode(new CASAdd(lhs, rhs));
}

CASNode* CASContext::getSub(CASNode* lhs, CASNode* rhs) {
    return addNode(new CASSub(lhs, rhs));
}

CASNode* CASContext::getMul(CASNode* lhs, CASNode* rhs) {
    return addNode(new CASMul(lhs, rhs));
}

CASPow* CASContext::getPow(CASNode* base, int exponent) {
    auto node = new CASPow(base, exponent);
    addNode(node);
    return node;
}

CASNode* CASContext::getNeg(CASNode* node) {
    return addNode(new CASNeg(node));
}

CASNode* CASContext::getCos(CASNode* node) {
    return addNode(new CASCos(node));
}

CASNode* CASContext::getSin(CASNode* node) {
    return addNode(new CASSin(node));
}

CASNode* CASContext::getMonomial(double coef, std::initializer_list<CASPow*> pows) {
    return addNode(new CASMonomial(coef, pows));
}

CASNode* CASContext::getMonomial(double coef, const std::vector<CASPow*>& pows) {
    return addNode(new CASMonomial(coef, pows));
}

CASNode* CASContext::getPolynomial(std::initializer_list<CASMonomial*> monomials) {
    return addNode(new CASPolynomial(monomials));
}

CASNode* CASContext::getPolynomial(const std::vector<CASMonomial*>& monomials) {
    return addNode(new CASPolynomial(monomials));
}

/* #endregion */

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

void CASPow::print(std::ostream& os) const {
    base->print(os);
    os << "**" << exponent;
}

void CASNeg::print(std::ostream& os) const {
    os << "-";
    node->print(os);
}

void CASCos::print(std::ostream& os) const {
    os << "cos(";
    node->print(os);
    os << ")";
}

void CASSin::print(std::ostream& os) const {
    os << "sin(";
    node->print(os);
    os << ")";
}

void CASMonomial::print(std::ostream& os) const {
    if (coefficient != 1.0)
        os << coefficient;
    for (auto* pow : pows)
        pow->print(os);
}

void CASPolynomial::print(std::ostream& os) const {
    auto length = monomials.size();
    if (length == 0)
        return;
    for (size_t i = 0; i < length-1; i++) {
        monomials[i]->print(os);
        os << " + ";
    }
    monomials[length-1]->print(os);
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
    else if (vL.isConstant && vL.value == 0.0)
        return { true, 0.0 };
    else if (vR.isConstant && vR.value == 0.0)
        return { true, 0.0 };
    return { false };
}

expr_value CASPow::getExprValue() const {
    auto vB = base->getExprValue();
    if (vB.isConstant)
        return { true, std::pow(vB.value, exponent) };
    return { false };
}

expr_value CASNeg::getExprValue() const {
    auto v = node->getExprValue();
    if (v.isConstant)
        return { true, -v.value };
    return { false };
}

expr_value CASCos::getExprValue() const {
    auto v = node->getExprValue();
    if (v.isConstant)
        return { true, std::cos(v.value) };
    return { false };
}

expr_value CASSin::getExprValue() const {
    auto v = node->getExprValue();
    if (v.isConstant)
        return { true, std::sin(v.value) };
    return { false };
}

expr_value CASMonomial::getExprValue() const {
    double value = coefficient;
    for (auto* pow : pows) {
        auto vPow = pow->getExprValue();
        if (!vPow.isConstant)
            return { false };
        value *= vPow.value;
    }
    return { true, value };
}

expr_value CASPolynomial::getExprValue() const {
    double value = 0.0;
    for (auto* monomial : monomials) {
        auto vM = monomial->getExprValue();
        if (!vM.isConstant)
            return { false };
        value += vM.value;
    }
    return { true, value };
}


CASNode* CASConstant::canonicalize(CASContext& ctx) const {
    return ctx.getMonomial(value, {});
}

CASNode* CASVariable::canonicalize(CASContext& ctx) const {
    return ctx.getMonomial(1.0, { ctx.getPow(ctx.getVariable(name), 1.0) });
}

CASNode* CASAdd::canonicalize(CASContext& ctx) const {
    auto newLHS = lhs->canonicalize(ctx);
    auto vL = newLHS->getExprValue();
    if (vL.isConstant && vL.value == 0.0)
        return rhs->canonicalize(ctx);

    auto newRHS = rhs->canonicalize(ctx);
    auto vR = newRHS->getExprValue();
    if (vR.isConstant && vR.value == 0.0)
        return newLHS;
    
    if (vL.isConstant && vR.isConstant)
        return ctx.getConstant(vL.value + vR.value);

    return ctx.getAdd(newLHS, newRHS);
}

CASNode* CASSub::canonicalize(CASContext& ctx) const {
    auto newLHS = lhs->canonicalize(ctx);
    auto vL = newLHS->getExprValue();
    if (vL.isConstant && vL.value == 0.0)
        return ctx.getNeg(rhs->canonicalize(ctx));

    auto newRHS = rhs->canonicalize(ctx);
    auto vR = newRHS->getExprValue();
    if (vR.isConstant && vR.value == 0.0)
        return newLHS;
    
    if (vL.isConstant && vR.isConstant)
        return ctx.getConstant(vL.value - vR.value);

    return ctx.getSub(newLHS, newRHS);
}

CASNode* CASMul::canonicalize(CASContext& ctx) const {
    auto newLHS = lhs->canonicalize(ctx);
    auto vL = newLHS->getExprValue();
    if (vL.isConstant) {
        if (vL.value == 0.0)
            return ctx.getConstant(0.0);
        if (vL.value == 1.0)
            return rhs->canonicalize(ctx);
        if (vL.value == -1.0)
            return ctx.getNeg(rhs->canonicalize(ctx));
    }

    auto newRHS = rhs->canonicalize(ctx);
    auto vR = newRHS->getExprValue();
    if (vR.isConstant) {
        if (vR.value == 0.0)
            return ctx.getConstant(0.0);
        if (vR.value == 1.0)
            return lhs->canonicalize(ctx);
        if (vR.value == -1.0)
            return ctx.getNeg(lhs->canonicalize(ctx));
    }
    if (vL.isConstant && vR.isConstant)
        return ctx.getConstant(vL.value * vR.value);

    return ctx.getMul(newLHS, newRHS);
}

CASNode* CASPow::canonicalize(CASContext& ctx) const {
    auto vB = base->getExprValue();
    if (vB.isConstant)
        return ctx.getConstant(std::pow(vB.value, exponent));
    return ctx.getPow(base, exponent);
}

CASNode* CASNeg::canonicalize(CASContext& ctx) const {
    auto vN = node->getExprValue();
    if (vN.isConstant)
        return ctx.getConstant(-vN.value);
    return ctx.getNeg(node);
}

CASNode* CASCos::canonicalize(CASContext& ctx) const {
    auto vN = node->getExprValue();
    if (vN.isConstant)
        return ctx.getConstant(std::cos(vN.value));
    return ctx.getCos(node);
}

CASNode* CASSin::canonicalize(CASContext& ctx) const {
    auto vN = node->getExprValue();
    if (vN.isConstant)
        return ctx.getConstant(std::sin(vN.value));
    return ctx.getSin(node);
}

CASNode* CASMonomial::canonicalize(CASContext& ctx) const {
    if (coefficient == 0.0)
        return ctx.getConstant(0.0);
    if (pows.empty())
        return ctx.getConstant(coefficient);
    return ctx.getMonomial(coefficient, pows);
}

CASNode* CASPolynomial::canonicalize(CASContext& ctx) const {
    return nullptr;
}

CASNode*
CASConstant::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getConstant(0.0);
}

CASNode*
CASVariable::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getConstant((var == name) ? 1.0 : 0.0);
}

CASNode*
CASAdd::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getAdd(lhs->derivative(var, ctx), rhs->derivative(var, ctx));
}

CASNode*
CASSub::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getSub(lhs->derivative(var, ctx), rhs->derivative(var, ctx));
}

CASNode*
CASMul::derivative(const std::string& var, CASContext& ctx) const {
    auto t1 = ctx.getMul(lhs->derivative(var, ctx), rhs);
    auto t2 = ctx.getMul(lhs, rhs->derivative(var, ctx));
    return ctx.getAdd(t1, t2);
}

CASNode*
CASPow::derivative(const std::string& var, CASContext& ctx) const {
    // (f(x)**n)' = n * f'(x) * f(x)**(n-1)
    auto t2 = ctx.getMul(base->derivative(var, ctx), exponent);
    auto t1 = ctx.getPow(base, exponent - 1.0);
    return ctx.getMul(t1, t2);
}

CASNode*
CASNeg::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getNeg(node->derivative(var, ctx));
}

CASNode*
CASCos::derivative(const std::string& var, CASContext& ctx) const {
    // cos(f(x))' = -f'(x) * sin(f(x))
    auto t1 = ctx.getNeg(node->derivative(var, ctx));
    auto t2 = ctx.getSin(node);
    return ctx.getMul(t1, t2);
}

CASNode*
CASSin::derivative(const std::string& var, CASContext& ctx) const {
    // sin(f(x))' = f'(x) * cos(f(x))
    auto t1 = node->derivative(var, ctx);
    auto t2 = ctx.getCos(node);
    return ctx.getMul(t1, t2);
}

CASNode*
CASMonomial::derivative(const std::string& var, CASContext& ctx) const {
    return nullptr;
}

CASNode*
CASPolynomial::derivative(const std::string& var, CASContext& ctx) const {
    return nullptr;
}
