#include "quench/parser.h"
#include <cassert>

using namespace quench;
using namespace quench::ast;

int Parser::readLine() {
    if (file.eof()) {
        file.close();
        return -1;
    }
    tokenVec.clear();
    do {
        std::getline(file, currentLine);
        lineNumber++;
        lineLength = currentLine.size();
    } while (!file.eof() && lineLength == 0);

    if (file.eof()) {
        file.close();
        return -1;
    }

    int col = 0;
    while (col < lineLength) {
        tokenVec.push_back(parseToken(col));
        col = tokenVec.back().colEnd;
    }
    tokenIt = tokenVec.cbegin();

    std::cerr << Color::CYAN_FG << lineNumber << " | " << currentLine << "\n";
    for (auto it = tokenVec.cbegin(); it != tokenVec.cend(); it++)
        std::cerr << "col " << it->colStart << "-" << it->colEnd << "  " << (*it) << "\n";
    std::cerr << Color::RESET;

    return tokenVec.size();
}

Token Parser::parseToken(int col) {
    if (col >= lineLength)
        // end of line
        return Token(TokenTy::EndOfLine, "", lineLength, lineLength+1);

    int curCol = col;
    char c = currentLine[curCol];
    while (c == ' ')
        c = currentLine[++curCol];
    int colStart = curCol;

    if (std::isdigit(c) || c == '.') {
        // numeric
        std::string str;
        while (true) {
            if (std::isdigit(c) || c == '.') {
                str += c;
                c = currentLine[++curCol];
                continue;
            }
            break;
        }
        return Token(TokenTy::Numeric, str, colStart, curCol);
    }
    if (std::isalpha(c)) {
        // identifier
        std::string str;
        while (true) {
            if (std::isalnum(c) || c == '_') {
                str += c;
                c = currentLine[++curCol];
                continue;
            }
            break;
        }
        if (str == "circuit")
            return Token(TokenTy::Circuit, "", colStart, curCol);
        else 
            return Token(TokenTy::Identifier, str, colStart, curCol);
    }

    char cnext = currentLine[curCol+1];
    // std::cerr << "next is " << next << "\n";
    switch (c) {
    // operators
    case '+':
        return Token(TokenTy::Add, "", colStart, colStart+1);
    case '-':
        return Token(TokenTy::Sub, "", colStart, colStart+1);
    case '*': // '**' or '*/' or '*'
        if (cnext == '*')
            return Token(TokenTy::Sub, "", colStart, colStart+2);
        if (cnext == '/')
            return Token(TokenTy::CommentEnd, "", colStart, colStart+2);
        return Token(TokenTy::Mul, "", colStart, colStart+1);
    case '/': // '//' or '/*' or '/'
        if (cnext == '/')
            return Token(TokenTy::Comment, "", colStart, colStart+2);
        if (cnext == '*')
            return Token(TokenTy::CommentStart, "", colStart, colStart+2);
        return Token(TokenTy::Div, "", colStart, colStart+1);
    case '=': // '==' or '='
        if (cnext == '=')
            return Token(TokenTy::EqualEqual, "", colStart, colStart+2);
        return Token(TokenTy::Equal, "", colStart, colStart+1);
    case '>': // '>=' or '>'
        if (cnext == '=')
            return Token(TokenTy::GreaterEqual, "", colStart, colStart+2);
        return Token(TokenTy::Greater, "", colStart, colStart+1);
    case '<': // '<=' or '<'
        if (cnext == '=')
            return Token(TokenTy::LessEqual, "", colStart, colStart+2);
        return Token(TokenTy::Less, "", colStart, colStart+1);
    // symbols
    case ',':
        return Token(TokenTy::Comma, "", colStart, colStart+1);
    case ';':
        return Token(TokenTy::Semicolon, "", colStart, colStart+1);
    case '(':
        return Token(TokenTy::L_RoundBraket, "", colStart, colStart+1);
    case ')':
        return Token(TokenTy::R_RoundBraket, "", colStart, colStart+1);
    case '[':
        return Token(TokenTy::L_SquareBraket, "", colStart, colStart+1);
    case ']':
        return Token(TokenTy::R_SquareBraket, "", colStart, colStart+1);
    case '{':
        return Token(TokenTy::L_CurlyBraket, "", colStart, colStart+1);
    case '}':
        return Token(TokenTy::R_CurlyBraket, "", colStart, colStart+1);
    case '\'':
        return Token(TokenTy::SingleQuote, "", colStart, colStart+1);
    case '\"':
        return Token(TokenTy::DoubleQuote, "", colStart, colStart+1);
    case '@':
        return Token(TokenTy::AtSymbol, "", colStart, colStart+1);
    case '%':
        return Token(TokenTy::Percent, "", colStart, colStart+1);
    case '#':
        return Token(TokenTy::Hash, "", colStart, colStart+1);
    case '\\':
        return Token(TokenTy::Backslash, "", colStart, colStart+1);
    case '\n':
        assert(false && "parsed LineFeed Token?");
        return Token(TokenTy::LineFeed, "", colStart, colStart+1);
    default:
        throwParserError("Unknown char " + std::to_string(c));
        assert(false && "Unknown char");
        return Token(TokenTy::Unknown, "", colStart, colStart+1);
    }
}

RootNode Parser::parse() {
    readLine();
    if (tokenIt->type == TokenTy::Circuit) {
        displayParserLog("ready to parse circuit");
        proceedWithType(TokenTy::Identifier);
        CircuitStmt circuit;
        circuit.name = tokenIt->str;
        proceedWithType(TokenTy::L_CurlyBraket, true);

        while (true) {
            proceed();
            if (tokenIt->type == TokenTy::Identifier) {
                GateChainStmt chain;
                while (true) {
                    chain.gates.push_back(_parseGateApply());
                    proceed();
                    if (tokenIt->type == TokenTy::AtSymbol) {
                        proceed();
                        continue;
                    }
                    if (tokenIt->type == TokenTy::Semicolon)
                        break;
                    throwParserError("Unexpected token type " + TokenTyToString(tokenIt->type)
                                    + " when expecting either AtSymbol or Semicolon");
                }
                circuit.addGateChain(chain);
                continue;
            }
            break;
        } 
        
        if (tokenIt->type != TokenTy::R_CurlyBraket) {
            throwParserError("Unexpected token " + tokenIt->to_string());
        }
        proceed(); // eat '}'
        displayParserLog("Parsed a circuit with " + std::to_string(circuit.stmts.size()) + " chains");
    }
    return {};
}

quench::quantum_gate::GateParameter Parser::_parseGateParameter() {
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        int i = convertCurTokenToInt();
        proceed();
        return { "%" + std::to_string(i) };
    }
    if (tokenIt->type == TokenTy::Numeric) {
        double real = convertCurTokenToFloat();
        if (optionalProceedWithType(TokenTy::Add)) {
            proceedWithType(TokenTy::Numeric);
            double imag = convertCurTokenToFloat();
            proceedWithType(TokenTy::Identifier);
            if (tokenIt->str != "i")
                throwParserError("Expect complex number to end with 'i'");
            proceed();
            return { std::complex<double>(real, imag) };
        }
        if (optionalProceedWithType(TokenTy::Identifier) && tokenIt->str == "i") {
            return { std::complex<double>(0.0, real) };
        }
        return { std::complex<double>(real, 0.0) };
    }

    throwParserError("Unable to parse gate parameter");
    assert(false && "Unreachable");
    return { std::complex<double>(0.0, 0.0) };
}

GateApplyStmt Parser::_parseGateApply() {
    assert(tokenIt->type == TokenTy::Identifier);
    GateApplyStmt gate(tokenIt->str);

    if (optionalProceedWithType(TokenTy::L_RoundBraket)) {
        if (optionalProceedWithType(TokenTy::Hash)) {
            proceedWithType(TokenTy::Numeric);
            gate.paramRefNumber = convertCurTokenToInt();
            proceedWithType(TokenTy::R_RoundBraket);
        }
        else if (optionalProceedWithType(TokenTy::Numeric)
                 || optionalProceedWithType(TokenTy::Percent)) {
            while (true) {
                gate.params.push_back(_parseGateParameter());
                if (tokenIt->type == TokenTy::Comma) {
                    proceed();
                    continue;
                }
                if (tokenIt->type == TokenTy::R_RoundBraket)
                    break;
            }
        }
    }

    // parse target qubits
    while (true) {
        if (optionalProceedWithType(TokenTy::Numeric)) {
            gate.qubits.push_back(convertCurTokenToInt());
            optionalProceedWithType(TokenTy::Comma);
            continue;
        }
        break;
    }
    
    if (gate.qubits.empty())
        throwParserError("Gate " + gate.name + " has no target");
    
    displayParserLog("Parsed gate " + gate.name + " with " +
                     std::to_string(gate.qubits.size()) + " targets");
    return gate;
}