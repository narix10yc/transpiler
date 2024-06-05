#include "quench/parser.h"
#include <cassert>

using namespace quench::ast;

int Parser::nextChar() {
    if (currentLine[column] == '\0') {
        if (file.eof()) {
            file.close();
            return -1;
        }
        // read new line
        while (true) {
            std::getline(file, currentLine);
            line++;
            if (file.eof()) {
                file.close();
                if (currentLine.empty())
                    return -1;
                break;
            }
            if (!currentLine.empty())
                break;
        }
        column = 0;
        return '\n';
    }
    return currentLine[column++];
}

int Parser::peekChar() {
    return currentLine[column];
}

Token parseCharsToToken(char c, char next) {
    switch (c) {
    // operators
    case '+': return { TokenTy::Add };
    case '-': return { TokenTy::Sub };
    case '*': { 
        if (next == '*') return { TokenTy::Pow }; // '**'
        if (next == '/') return { TokenTy::CommentEnd }; // '*/'
        return { TokenTy::Mul };
    }
    case '/': {
        if (next == '/') return { TokenTy::Comment }; // '//'
        if (next == '*') return { TokenTy::CommentStart }; // '/*'
        return { TokenTy::Div };
    }
    case '=': return { (next == '=') ? TokenTy::EqualEqual : TokenTy::Equal };
    case '>': return { (next == '=') ? TokenTy::GreaterEqual : TokenTy::Greater };
    case '<': return { (next == '=') ? TokenTy::LessEqual : TokenTy::Less };
    // symbols
    case ',': return { TokenTy::Comma };
    case ';': return { TokenTy::Semicolon };
    case '(': return { TokenTy::L_RoundBraket };
    case ')': return { TokenTy::R_RoundBraket };
    case '[': return { TokenTy::L_SquareBraket };
    case ']': return { TokenTy::R_SquareBraket };
    case '{': return { TokenTy::L_CurlyBraket };
    case '}': return { TokenTy::R_CurlyBraket };
    case '\'': return { TokenTy::SingleQuote };
    case '\"': return { TokenTy::DoubleQuote };
    case '@': return { TokenTy::AtSymbol };
    case '%': return { TokenTy::Percent };
    case '#': return { TokenTy::Hash };
    case '\\': return { TokenTy::Backslash };

    case '\n': return { TokenTy::LineFeed };
    default: return { TokenTy::Unknown };
    }
}

bool Parser::proceed() {
    curToken = nextToken;
    if (curToken.type == TokenTy::Eof)
        return false;

    int c;
    // skip white space
    while ((c = nextChar()) == ' ');

    if (c < 0)
        nextToken = { TokenTy::Eof };
    else if (std::isdigit(c) || c == '.') {
        // numeric
        std::string str {static_cast<char>(c)};
        c = peekChar();
        while (isdigit(c) || c == '.') {
            str += static_cast<char>(c);
            nextChar();
            c = peekChar();
        }
        nextToken = { TokenTy::Numeric, str };
    }
    else if (std::isalpha(c)) {
        // identifier
        std::string str {static_cast<char>(c)};
        c = peekChar();
        while (isalnum(c) || c == '_') {
            str += static_cast<char>(c);
            nextChar();
            c = peekChar();
        }
        if (str == "circuit")
            nextToken = { TokenTy::Circuit };
        else 
            nextToken = { TokenTy::Identifier, str };
    }
    else {
        int next = peekChar();
        assert(c >= 0 && c <= 255);
        nextToken = parseCharsToToken(c, next);

        if (nextToken.type == TokenTy::Unknown) {
            std::string errMsg("Unknown char '");
            errMsg += c; errMsg += '\'';
            displayParserError(errMsg);
            assert(false && "Unknown char");
            return false;
        }
        return true;
    }
    return true;
}

void Parser::skipRestOfLine() {
    column = currentLine.size();
}

std::unique_ptr<GateApplyStmt> Parser::parseGateApplyStmt_() {
    return nullptr;
}

std::unique_ptr<CircuitStmt> Parser::parseCircuitStmt_() {
    return nullptr;
}


std::unique_ptr<RootNode> Parser::parse() {
    proceed();
    auto root = std::make_unique<RootNode>();
    while (proceed()) {
        if (curToken.type == TokenTy::Comment) {
            skipRestOfLine();
            continue;
        }
        std::cerr << curToken << " ";
    }
    return root;
}