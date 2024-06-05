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
    std::cerr << "curToken: " << curToken << "\n";
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
    assert(curToken.type == TokenTy::Identifier);

    errorMsgStart = "GateApplyStmt";
    auto gate = std::make_unique<GateApplyStmt>(curToken.str);

    // parameters
    if (proceedWithType(TokenTy::L_RoundBraket, false)) {
        displayParserWarning("Parsing parameters in gate is not implemented yet");
        if (!proceedWithType(TokenTy::R_RoundBraket)) return nullptr;
    }

    while (proceedWithType(TokenTy::Numeric, false)) {
        int qubit;
        try {
            qubit = std::stoi(curToken.str);
        } catch (...) {
            displayParserError("Cannot parse target qubit '" + curToken.str + "'");
            return nullptr;
        }
        gate->qubits.push_back(qubit);
    }

    if (gate->qubits.empty())
        displayParserWarning("Parsed a gate with no target");
    else
        proceed(); // eat the last target qubit

    return gate;
}

std::unique_ptr<CircuitStmt> Parser::parseCircuitStmt_() {
    assert(curToken.type == TokenTy::Circuit);
    errorMsgStart = "CircuitStmt";

    auto circuit = std::make_unique<CircuitStmt>();    
    // number of qubits
    if (proceedWithType(TokenTy::Less, false)) {
        if (!proceedWithType(TokenTy::Numeric)) return nullptr;
        try {
            circuit->nqubits = std::stoi(curToken.str);
        } catch (std::invalid_argument) {
            displayParserError("Cannot parse number of qubits with input '" + curToken.str + "'");
            return nullptr;
        } catch (std::out_of_range) {
            displayParserError("Number of qubits out of range '" + curToken.str + "'");
            return nullptr;
        }
        if (!proceedWithType(TokenTy::Greater)) return nullptr;
    }

    // name
    if (!proceedWithType(TokenTy::Identifier)) return nullptr;
    circuit->name = curToken.str;

    // parameters
    if (proceedWithType(TokenTy::L_RoundBraket, false)) {
        displayParserWarning("Parsing parameters in circuit is not implemented yet");
        if (!proceedWithType(TokenTy::R_RoundBraket)) return nullptr;
    }

    // body (gates)
    if (!proceedWithType(TokenTy::L_CurlyBraket)) return nullptr;
    proceed(); // eat '{'

    std::unique_ptr<Statement> stmt = nullptr;
    while ((stmt = parseStatement_()) != nullptr) {
        auto gate = dynamic_cast<GateApplyStmt*>(stmt.get());
        if (gate == nullptr) {
            displayParserError("Only gates are allowed in circuit (for now)...");
            return nullptr;
        }
        circuit->addGate(std::make_unique<GateApplyStmt>(*gate));
    }
    if (!proceedWithType(TokenTy::R_CurlyBraket)) return nullptr;

    proceed(); // eat '}'
    return circuit;
}

std::unique_ptr<Statement> Parser::parseStatement_() {
    while (true) {
        if (curToken.type == TokenTy::Eof) {
            displayParserWarning("Got EoF when trying start a Statement");
            return nullptr;
        }
        if (curToken.type == TokenTy::Comment) {
            skipRestOfLine();
            continue;
        }
        if (curToken.type == TokenTy::LineFeed ||
                                curToken.type == TokenTy::Semicolon) {
            proceed();
            continue;
        }

        break;
    }

    if (curToken.type == TokenTy::Circuit)
        return parseCircuitStmt_();
    return parseGateApplyStmt_();
}


std::unique_ptr<RootNode> Parser::parse() {
    proceed(); proceed();
    auto root = std::make_unique<RootNode>();
    std::unique_ptr<Statement> stmt = nullptr;
    while ((stmt = parseStatement_()) != nullptr) {
        root->stmts.push_back(std::move(stmt));
    }
    return root;
}