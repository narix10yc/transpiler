#include "parser.h"


std::unique_ptr<ast::Expression> Parser::parseExpr() {
    logDebug(2, "Expression: ready to parse; curToken = " + curToken.ToString());
    
    auto lhs = parsePrimaryExpr();
    if (!lhs) {
        logError("Failed to parse LHS when attempting to parse an expression");
        return nullptr;
    }
    logDebug(3, "Expression: Parsed LHS " + lhs->ToString());
    
    nextToken(); // eat lhs expression
    BinaryOp binop = curToken.toBinaryOp();
    if (binop == BinaryOp::None) { // Not a binop
        logDebug(3, "Not a binop, return lhs");
        return lhs;
    }

    return parseExprRHS(binop, std::move(lhs));
}

std::unique_ptr<ast::Expression> 
Parser::parseExprRHS(BinaryOp lhsBinop, std::unique_ptr<ast::Expression> &&lhs) {
    logDebug(3, "Expression: ready to parse RHS");
    auto rhs = parsePrimaryExpr();
    if (!rhs) {
        logError("Missing RHS of a binary expression");
        return std::move(lhs);
    }
    logDebug(3, "Expression: Parsed RHS " + rhs->ToString());

    nextToken(); // eat rhs expression
    BinaryOp binop = curToken.toBinaryOp();
    if (binop == BinaryOp::None) {
        auto ret = std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs), std::move(rhs));
        logDebug(3, "Expression: No more binop, returning " + ret->ToString());
        return std::move(ret);
    }

    // Encounter another binop
    if (getBinopPrecedence(binop) > getBinopPrecedence(lhsBinop)) {
        auto newRHS = parseExprRHS(binop, std::move(rhs));
        return std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs), std::move(newRHS));
    } 
    else {
        auto newLHS = std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs), std::move(rhs));
        return parseExprRHS(binop, std::move(newLHS));
    }
}


std::unique_ptr<ast::Expression> Parser::parsePrimaryExpr() {
    if (curToken.type == TokenTy::Numeric) {
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
    double numeric = std::stod(curToken.str);
    auto expr = std::make_unique<ast::NumericExpr>(numeric);
    nextToken();
    return expr;
}

std::unique_ptr<ast::VariableExpr> Parser::parseVariableExpr() {
    auto expr = std::make_unique<ast::VariableExpr>(curToken.str);
    nextToken();
    return expr;
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
