#include "saot/Polynomial.h"
#include <cassert>

using namespace saot::polynomial;

Node* Numerics::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.getNumerics(0.0);
}

Node* Variable::derivative(const std::string& var, CASContext& ctx) const {
    if (name == var)
        return ctx.getNumerics(1.0);
    return ctx.getNumerics(0.0);
}

Node* CosExpr::derivative(const std::string& var, CASContext& ctx) const {
    // -sin(node) * node'
    auto* t1 = ctx.createSin(node);
    auto* t2 = node->derivative(var, ctx);
    return ctx.createMul(ctx.getNumerics(-1.0), ctx.createMul(t1, t2));
}

Node* SinExpr::derivative(const std::string& var, CASContext& ctx) const {
    // cos(node) * node'
    return ctx.createMul(ctx.createCos(node), node->derivative(var, ctx));
}

Node* AddExpr::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.createAdd(lhs->derivative(var, ctx), rhs->derivative(var, ctx));
}

Node* SubExpr::derivative(const std::string& var, CASContext& ctx) const {
    return ctx.createSub(lhs->derivative(var, ctx), rhs->derivative(var, ctx));
}

Node* MulExpr::derivative(const std::string& var, CASContext& ctx) const {
    auto* t1 = ctx.createMul(lhs->derivative(var, ctx), rhs);
    auto* t2 = ctx.createMul(lhs, rhs->derivative(var, ctx));
    return ctx.createAdd(t1, t2);
}


Node* CosExpr::simplify(CASContext& ctx) {
    auto* s_node = node->simplify(ctx);
    if (auto* num_node = dynamic_cast<Numerics*>(s_node))
        return ctx.getNumerics(std::cos(num_node->value));

    if (s_node == node)
        return this;
    return ctx.createCos(s_node);
}

Node* SinExpr::simplify(CASContext& ctx) {
    auto* s_node = node->simplify(ctx);
    if (auto* num_node = dynamic_cast<Numerics*>(s_node))
        return ctx.getNumerics(std::sin(num_node->value));
    
    if (s_node == node)
        return this;
    return ctx.createSin(s_node);
}

Node* AddExpr::simplify(CASContext& ctx) {
    auto* s_lhs = lhs->simplify(ctx);
    auto* s_rhs = rhs->simplify(ctx);
    auto* num_lhs = dynamic_cast<Numerics*>(s_lhs);
    auto* num_rhs = dynamic_cast<Numerics*>(s_rhs);
    if (num_lhs && num_rhs)
        return ctx.getNumerics(num_lhs->value + num_rhs->value);
    if (lhs == rhs)
        return ctx.createMul(ctx.getNumerics(2.0), s_lhs);
    
    if (s_lhs == lhs && s_rhs == rhs)
        return this;
    return ctx.createAdd(s_lhs, s_rhs);
}

Node* SubExpr::simplify(CASContext& ctx) {
    if (lhs == rhs)
        return ctx.getNumerics(0.0);

    auto* s_lhs = lhs->simplify(ctx);
    auto* s_rhs = rhs->simplify(ctx);
    auto* num_lhs = dynamic_cast<Numerics*>(s_lhs);
    auto* num_rhs = dynamic_cast<Numerics*>(s_rhs);
    if (num_lhs && num_rhs)
        return ctx.getNumerics(num_lhs->value + num_rhs->value);

    if (s_lhs == lhs && s_rhs == rhs)
        return this;
    return ctx.createAdd(s_lhs, s_rhs);
}

Node* MulExpr::simplify(CASContext& ctx) {
    auto* s_lhs = lhs->simplify(ctx);
    auto* s_rhs = rhs->simplify(ctx);
    auto* num_lhs = dynamic_cast<Numerics*>(s_lhs);
    if (num_lhs && num_lhs->value == 0.0)
        return num_lhs;
    auto* num_rhs = dynamic_cast<Numerics*>(s_rhs);
    if (num_rhs && num_rhs->value == 0.0)
        return num_rhs;
    if (num_lhs && num_rhs)
        return ctx.getNumerics(num_lhs->value * num_rhs->value);
    
    if (s_lhs == lhs && s_rhs == rhs)
        return this;
    return ctx.createMul(s_lhs, s_rhs);
}

