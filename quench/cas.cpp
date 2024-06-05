#include "quench/ast.h"

using namespace quench::ast;

PolynomialExpr ConstantExpr::toPolynomial() const {
    return { MonomialExpr(std::make_shared<ConstantExpr>(*this)) };
}

PolynomialExpr VariableExpr::toPolynomial() const {
    return { MonomialExpr(std::make_shared<VariableExpr>(*this)) };
}

PolynomialExpr CosineExpr::toPolynomial() const {
    return { MonomialExpr(std::make_shared<CosineExpr>(*this)) };
}

PolynomialExpr SineExpr::toPolynomial() const {
    return { MonomialExpr(std::make_shared<SineExpr>(*this)) };
}

PolynomialExpr PowerExpr::toPolynomial() const {
    return { MonomialExpr({*this}) };
}

PolynomialExpr MonomialExpr::toPolynomial() const {
    return { *this };
}