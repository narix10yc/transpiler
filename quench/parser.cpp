#include "quench/parser.h"
#include <cassert>

using namespace quench::cas;
using namespace quench::ast;

int Parser::readLine() {
    if (file.eof()) {
        file.close();
        return -1;
    }
    std::getline(file, currentLine);
    line++;
    return currentLine.size();
}

int Parser::nextChar() {
    column++;
    if (column == currentLine.size())
        return '\n';
    if (column > currentLine.size()) {
        int flag = 0;
        while ((flag = readLine()) == 0) {}
        if (flag < 0)
            return -1;
        column = 0;
        return currentLine[0];
    }
    return currentLine[column];
}

int Parser::peekChar() {
    if (column >= currentLine.size())
        return '\0';
    return currentLine[column+1];
}

bool Parser::proceed() {
    curToken = nextToken;
    if (curToken.type == TokenTy::Eof)
        return false;

    int c;
    // skip white space
    while ((c = nextChar()) == ' ');
    
    // std::cerr << "c is " << c << "\n";
    if (c < 0) {
        nextToken = { TokenTy::Eof };
    }
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
        // std::cerr << "next is " << next << "\n";
        assert(c >= 0 && c <= 255);
        switch (c) {
        // operators
        case '+': nextToken = { TokenTy::Add }; break;
        case '-': nextToken = { TokenTy::Sub }; break;
        case '*': // '**' or '*/' or '*'
            if (next == '*') {
                column++;
                nextToken = { TokenTy::Pow };
            } else if (next == '/') {
                column++;
                nextToken = { TokenTy::CommentEnd };
            } else nextToken = { TokenTy::Mul };
            break;
        case '/': { // '//' or '/*' or '/'
            if (next == '/') {
                column++;
                nextToken = { TokenTy::Comment };
            } else if (next == '*') {
                column++;
                nextToken = { TokenTy::CommentStart };
            } else nextToken = { TokenTy::Div };
            break;
        }
        case '=': // '==' or '='
            if (next == '=') {
                column++;
                nextToken = { TokenTy::EqualEqual };
            } else nextToken = { TokenTy::Equal };
            break;
        case '>': // '>=' or '>'
            if (next == '=') {
                column++;
                nextToken = { TokenTy::GreaterEqual };
            } else nextToken = { TokenTy::Greater };
            break;
        case '<': // '<=' or '<'
            if (next == '=') {
                column++;
                nextToken = { TokenTy::LessEqual };
            } else nextToken = { TokenTy::Less };
            break;
        // symbols
        case ',': nextToken = { TokenTy::Comma }; break;
        case ';': nextToken = { TokenTy::Semicolon }; break;
        case '(': nextToken = { TokenTy::L_RoundBraket }; break;
        case ')': nextToken = { TokenTy::R_RoundBraket }; break;
        case '[': nextToken = { TokenTy::L_SquareBraket }; break;
        case ']': nextToken = { TokenTy::R_SquareBraket }; break;
        case '{': nextToken = { TokenTy::L_CurlyBraket }; break;
        case '}': nextToken = { TokenTy::R_CurlyBraket }; break;
        case '\'': nextToken = { TokenTy::SingleQuote }; break;
        case '\"': nextToken = { TokenTy::DoubleQuote }; break;
        case '@': nextToken = { TokenTy::AtSymbol }; break;
        case '%': nextToken = { TokenTy::Percent }; break;
        case '#': nextToken = { TokenTy::Hash }; break;
        case '\\': nextToken = { TokenTy::Backslash }; break;

        case '\n': nextToken = { TokenTy::LineFeed }; break;
        default:
            nextToken = { TokenTy::Unknown };
            throwParserError("Unknown char " + std::to_string(c));
            assert(false && "Unknown char");
            return false;
        }
    }
    std::cerr << CYAN_FG << "(" << line << ":" << column << ")"
              << " proceed: " << curToken << " " << nextToken << "\n" << RESET;
    return true;
}

Polynomial Parser::parsePolynomial_() {
    if (curToken.type == TokenTy::Numeric) {
        return {{convertCurTokenToFloat(), {}}};
    }
    displayParserWarning("Only Numerics is supported in parsePolynomial yet");
    return {};
}

GateApplyStmt Parser::parseGateApplyStmt_() {
    assert(curToken.type == TokenTy::Identifier);

    GateApplyStmt gate{curToken.str};

    // parameters
    if (proceedWithType(TokenTy::L_RoundBraket, false)) {
        proceedWithType(TokenTy::Hash);
        proceedWithType(TokenTy::Numeric);
        gate.paramRefNumber = convertCurTokenToInt();
        proceedWithType(TokenTy::R_RoundBraket);
    }

    proceed();
    while (true) {
        gate.qubits.push_back(convertCurTokenToInt());
        proceed();
        if (curToken.type != TokenTy::Numeric)
            break;
    }
    if (gate.qubits.empty())
        throwParserError("GateApply: Parsed a gate with no target qubit");
    
    displayParserLog("Parsed gate '" + gate.name + "'");
    return gate;
}

GateChainStmt Parser::parseGateChainStmt_() {
    assert(curToken.type == TokenTy::Identifier);;

    GateChainStmt chain{};

    while (true) {
        chain.gates.push_back(parseGateApplyStmt_());
        skipLineFeeds();
        if (curToken.type == TokenTy::AtSymbol) {
            proceed(); continue;
        }
        if (curToken.type == TokenTy::Semicolon) {
            proceed();
            break;
        }
        throwParserError("GateChain: Unrecognized curToken type");
    }
    displayParserLog("Parsed a gate chain with " + std::to_string(chain.gates.size()) + " gates");
    return chain;
}

CircuitStmt Parser::parseCircuitStmt_() {
    assert(curToken.type == TokenTy::Circuit);

    CircuitStmt circuit;
    // number of qubits
    if (proceedWithType(TokenTy::Less, false)) {
        proceedWithType(TokenTy::Numeric);
        circuit.nqubits = convertCurTokenToInt();
        proceedWithType(TokenTy::Greater);
    }

    // name
    proceedWithType(TokenTy::Identifier);
    circuit.name = curToken.str;

    // parameters
    if (proceedWithType(TokenTy::L_RoundBraket, false)) {
        displayParserWarning("Parsing parameters in circuit is not implemented yet");
        proceedWithType(TokenTy::R_RoundBraket);
    }

    // body (gates)
    proceedWithType(TokenTy::L_CurlyBraket);
    proceed(); skipLineFeeds();
    while (true) {
        if (curToken.type == TokenTy::R_CurlyBraket) {
            proceed();
            break;
        }
        circuit.addGateChain(parseGateChainStmt_());
        skipLineFeeds();
        continue;
    }
    displayParserLog("Parsed circuit '" + circuit.name + "'");
    return circuit;
}

ParameterDefStmt Parser::parseParameterDefStmt_() {
    assert(curToken.type == TokenTy::Hash);

    int refNumber;
    proceedWithType(TokenTy::Numeric);
    refNumber = convertCurTokenToInt();
    proceedWithType(TokenTy::Equal);
    proceedWithType(TokenTy::L_CurlyBraket);

    ParameterDefStmt defStmt{refNumber};
    while (proceed()) {
        Polynomial real, imag;
        real = parsePolynomial_();
        proceedWithType(TokenTy::Comma);
        proceed();
        imag = parsePolynomial_();
        proceed();
        defStmt.matrix.matrix.push_back({real, imag});
        if (curToken.type == TokenTy::Comma)
            continue;
        if (curToken.type == TokenTy::R_CurlyBraket)
            { proceed(); break; }
        throwParserError("ParameterDef: Unrecognized tokenType");
    }

    // update number of qubits;
    int s = defStmt.matrix.updateNQubits();
    if (s < 0)
        throwParserError("ParameterDef: Failed to update matrix");

    std::stringstream ss;
    ss << "Parsed a ParameterDefStmt with " << s << "x" << s << " matrix";
    displayParserLog(ss.str());
    return defStmt;
}

bool Parser::parseStatement_(RootNode& root) {
    while (true) {
        if (curToken.type == TokenTy::Eof) {
            return false;
        }
        if (curToken.type == TokenTy::Comment) {
            column = currentLine.size();
            nextToken = { TokenTy::LineFeed };
            proceed();
            continue;
        }
        // skip \n and ';'
        if (curToken.type == TokenTy::LineFeed
            || curToken.type == TokenTy::Semicolon
            || curToken.type == TokenTy::Start) {
            proceed();
            continue;
        }
        break;
    }

    if (curToken.type == TokenTy::Circuit) {
        root.addCircuit(std::make_unique<CircuitStmt>(parseCircuitStmt_()));
    }
    else if (curToken.type == TokenTy::Hash) {
        auto defStmt = parseParameterDefStmt_();
        if (!root.addParameterDef(defStmt)) {
            std::stringstream ss;
            ss << "Parameter with ref number " << defStmt.refNumber
               << " has already been defined";
            throwParserError(ss.str());
        }
    }
    else {
        throwParserError("Statement: Unrecognized curToken type");
    }
    return true;
}

RootNode Parser::parse() {
    RootNode root;
    while (parseStatement_(root)) {}
    if (curToken.type != TokenTy::Eof)
        displayParserWarning("end of parsing, but curToken is not EoF?");
    else
        displayParserLog("Reached EoF");
    return root;
}