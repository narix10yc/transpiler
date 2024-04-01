#include "parser.h"

using namespace openqasm;

std::unique_ptr<ast::Expression> Parser::parseExpr() {
    logDebug(2, "Expression: ready to parse; curToken = " + curToken.toString());
    
    auto lhs = parsePrimaryExpr();
    if (!lhs) {
        logError("Expression: unable to parse lhs");
        return nullptr;
    }
    logDebug(3, "Expression: parsed lhs " + lhs->toString());
    
    BinaryOp binop = curToken.toBinaryOp();
    if (binop == BinaryOp::None) { // Not a binop
        logDebug(3, "Not a binop, return lhs");
        return lhs;
    }

    nextToken(); // eat the binop
    return parseExprRHS(binop, std::move(lhs));
}

std::unique_ptr<ast::Expression> 
Parser::parseExprRHS(BinaryOp lhsBinop, std::unique_ptr<ast::Expression> &&lhs) {
    logDebug(3, "Expression: ready to parse rhs");
    auto rhs = parsePrimaryExpr();
    if (!rhs) {
        logError("Expression: unable to parse rhs");
        return nullptr;
    }
    logDebug(3, "Expression: parsed rhs " + rhs->toString());

    BinaryOp binop = curToken.toBinaryOp();
    if (binop == BinaryOp::None) {
        logDebug(3, "Expression: No more binop, return");
        auto ret = std::make_unique<ast::BinaryExpr>
            (lhsBinop, std::move(lhs), std::move(rhs));
        return std::move(ret);
    }

    // Encounter another binop
    nextToken(); // eat the binop
    if (getBinopPrecedence(binop) > getBinopPrecedence(lhsBinop)) {
        auto newRHS = parseExprRHS(binop, std::move(rhs));
        return std::make_unique<ast::BinaryExpr>
            (lhsBinop, std::move(lhs), std::move(newRHS));
    } 
    else {
        auto newLHS = std::make_unique<ast::BinaryExpr>
            (lhsBinop, std::move(lhs), std::move(rhs));
        return parseExprRHS(binop, std::move(newLHS));
    }
}

std::unique_ptr<ast::Expression> Parser::parsePrimaryExpr() {
    if (curToken.type == TokenTy::Numeric || curToken.type == TokenTy::Sub) {
        return parseNumericExpr();
    }
    else if (curToken.type == TokenTy::Identifier) {
        return parseVariableExpr();
    }
    else if (curToken.type == TokenTy::L_RoundBraket) {
        return parseParenExpr();
    }
    // else
    logError("Unknown token when expecting a primary expression");
    return nullptr;   
}

std::unique_ptr<ast::NumericExpr> Parser::parseNumericExpr() {
    bool negFlag = false;
    if (curToken.type == TokenTy::Sub) {
        negFlag = true;
        nextToken();
    }
    if (curToken.type != TokenTy::Numeric) {
        logError("Expect numerics when parsing Numerics Expression");
        return nullptr;
    }
    double numeric = std::stod(curToken.str);
    auto expr = std::make_unique<ast::NumericExpr>(negFlag ? -numeric : numeric);
    nextToken();
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parseVariableExpr() {
    auto name = curToken.str;
    nextToken(); // eat the identifier

    if (curToken.type == TokenTy::L_RoundBraket) {
        // funcCall
        logError("NOT IMPLEMENTED");
    }
    else if (curToken.type == TokenTy::L_SquareBraket) {
        // subscript
        nextToken(); // eat '['
        auto numericExpr = parseNumericExpr();
        if (!numericExpr) {
            logError("Expect index");
            return nullptr;
        }
        int index = static_cast<int>(numericExpr->getValue());
        if (curToken.type != TokenTy::R_SquareBraket) {
            logError("Subscript Expression: expect ']'");
            return nullptr;
        }

        nextToken(); // eat ']'
        return std::make_unique<ast::SubscriptExpr>(name, index);
    }

    return std::make_unique<ast::VariableExpr>(name);
}

std::unique_ptr<ast::Expression> Parser::parseParenExpr() {
    nextToken(); // Eat '('
    auto expr = parseExpr();
    if (!expr) {
        return nullptr;
    }
    if (curToken.type != TokenTy::R_RoundBraket) {
        logError("Expected ')'");
        return nullptr;
    }
    nextToken(); // Eat ')'
    return expr;
}
