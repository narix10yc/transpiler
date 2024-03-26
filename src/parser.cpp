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

    if (curToken.type == TokenTy::If) {
        stmt = parseIfThenElseStmt();
    } 
    else if (curToken.type == TokenTy::Openqasm) {
        stmt = parseVersionStmt();
    }

    skipSeparators();

    return std::move(stmt);
}

std::unique_ptr<ast::VersionStmt> Parser::parseVersionStmt() {
    return nullptr;
}