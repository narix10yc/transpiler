#include "parser.h"


void Parser::parse() {
    if (!lexer->checkFileOpen()) {
        logError("Unable to open file");
        return;
    }
    nextToken();

    while (curToken.type != TokenTy::Eof) {
        auto stmt = parseStmt();
        if (!stmt) {
            logError("Failed to parse stmt");
            return;
        }
        logDebug(1, "Parsed Stmt " + stmt->ToString());
        root->addStmt(std::move(stmt));
    }

    logDebug(1, "Reached the end of file");
}

std::unique_ptr<ast::Statement> Parser::parseStmt() {
    std::unique_ptr<ast::Statement> stmt;
    skipSeparators();

    if (curToken.type == TokenTy::If) {
        stmt = parseIfThenElseStmt();
    } 
    else if (curToken.type == TokenTy::Openqasm) {
        logDebug(2, "ready to parse openqasm");
        stmt = parseVersionStmt();
    }
    else if (curToken.type == TokenTy::Qreg) {
        logDebug(2, "ready to parse qreg");
        stmt = parseQRegStmt();
    }
    else if (curToken.type == TokenTy::Creg) {
        logDebug(2, "ready to parse creg");
        stmt = parseCRegStmt();
    }
    else {
        logError("Encountered TokenType " + tokenTypeToString(curToken.type));
    }

    skipSeparators();

    return std::move(stmt);
}

std::unique_ptr<ast::VersionStmt> Parser::parseVersionStmt() {
    nextToken();
    if (curToken.type != TokenTy::Numeric) {
        logError("Expect version string to be a numerics");
        return nullptr;
    }
    auto value = curToken.str;
    nextToken();
    return std::make_unique<ast::VersionStmt>(value);
}

std::unique_ptr<ast::QRegStmt> Parser::parseQRegStmt() {
    if (!expectNextToken(TokenTy::Identifier)) 
        return nullptr;

    auto name = curToken.str;
    if (!expectNextToken(TokenTy::L_SquareBraket))
        return nullptr;
    if (!expectNextToken(TokenTy::Numeric))
        return nullptr;

    int size = std::stoi(curToken.str);
    if (!expectNextToken(TokenTy::R_SquareBraket))
        return nullptr;

    auto stmt = std::make_unique<ast::QRegStmt>(name, size);
    nextToken();
    return std::move(stmt);
}

std::unique_ptr<ast::CRegStmt> Parser::parseCRegStmt() {
    if (!expectNextToken(TokenTy::Identifier)) 
        return nullptr;
        
    auto name = curToken.str;
    if (!expectNextToken(TokenTy::L_SquareBraket))
        return nullptr;
    if (!expectNextToken(TokenTy::Numeric))
        return nullptr;

    int size = std::stoi(curToken.str);
    if (!expectNextToken(TokenTy::R_SquareBraket))
        return nullptr;
    nextToken();

    auto stmt = std::make_unique<ast::CRegStmt>(name, size);
    nextToken();
    return std::move(stmt);
}