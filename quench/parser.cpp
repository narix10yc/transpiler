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
    // std::cerr << CYAN_FG << "(" << line << ":" << column << ")"
            //   << " before proceed: " << curToken << "; " << nextToken << "\n" << RESET;

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
            displayParserError("Unknown char " + std::to_string(c));
            assert(false && "Unknown char");
            return false;
        }
    }
    std::cerr << CYAN_FG << "(" << line << ":" << column << ")"
              << " proceed: " << curToken << "; " << nextToken << "\n" << RESET;
    return true;
}


Polynomial Parser::parsePolynomial_() {
    if (curToken.type == TokenTy::Numeric) {
        double value;
        try {
            value = std::stod(curToken.str);
        } catch (...) {
            displayParserError("Cannot parse numerics '" + curToken.str + "'");
            return {};
        }
        return {{value, {}}};
    }
    displayParserWarning("Only Numerics is supported in parsePolynomial yet");
    return {};
}

std::unique_ptr<GateApplyStmt> Parser::parseGateApplyStmt_() {
    assert(curToken.type == TokenTy::Identifier);
    errorMsgStart = "GateApplyStmt";

    auto gate = std::make_unique<GateApplyStmt>(curToken.str);
    displayParserLog("start parsing a gate with name " + curToken.str);

    // parameters
    if (proceedWithType(TokenTy::L_RoundBraket, false)) {
        if (!proceedWithType(TokenTy::Hash)) return nullptr;
        if (!proceedWithType(TokenTy::Numeric)) return nullptr;
        try {
            gate->paramRefNumber = std::stoi(curToken.str);
        } catch (...) {
            displayParserError("Cannot parse target parameter reference '" + curToken.str + "'");
            return nullptr;
        }
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

    displayParserLog("parsed a gate");
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

    while (proceed()) {
        errorMsgStart = "CircuitStmt";
        std::cerr << "Hello World! curToken is " << curToken << "\n";
        if (curToken.type == TokenTy::R_CurlyBraket) {
            proceed(); // eat '}'
            displayParserLog("parsed a circuit");
            return circuit;
        }
        if (curToken.type == TokenTy::Semicolon
                        || curToken.type == TokenTy::LineFeed) {
            continue;
        }
        if (curToken.type == TokenTy::Identifier) {
            circuit->addGate(parseGateApplyStmt_());
            continue;
        }
        displayParserError("Unrecognized curToken when expecting "
                           "an identifier to parse a gate");
        return nullptr;
    }
    return nullptr;
}

std::unique_ptr<ParameterDefStmt> Parser::parseParameterDefStmt_() {
    assert(curToken.type == TokenTy::Hash);
    errorMsgStart = "ParameterDefStmt";

    int refNumber;
    if (!proceedWithType(TokenTy::Numeric)) return nullptr;
    try {
        refNumber = std::stoi(curToken.str);
    } catch (...) {
        displayParserError("Cannot parse target qubit '" + curToken.str + "'");
        return nullptr;
    }

    if (!proceedWithType(TokenTy::Equal)) return nullptr;
    if (!proceedWithType(TokenTy::L_CurlyBraket)) return nullptr;

    auto defStmt = std::make_unique<ParameterDefStmt>(refNumber);
    while (proceed()) {
        Polynomial real, imag;
        real = parsePolynomial_();
        if (!proceedWithType(TokenTy::Comma)) return nullptr;
        proceed();
        imag = parsePolynomial_();
        proceed();
        defStmt->matrix.data.push_back({real, imag});
        if (curToken.type == TokenTy::Comma)
            continue;
        if (curToken.type == TokenTy::R_CurlyBraket)
            { proceed(); break; }
        displayParserError("Unrecognized tokenType");
        return nullptr;
    }

    // update number of qubits;
    int s = defStmt->matrix.updateSize();
    if (s < 0) {
        displayParserError("Failed to updateMatrix due to size");
        return defStmt;
    }
    if (s == 0) {
        displayParserWarning("Parsed a ParamDefStmt with 0 param?");
        return defStmt;
    }
    if (s == 1) {
        displayParserLog("Parsed a 1-qubit ParamDefStmt");
        defStmt->nqubits = 1;
        return defStmt;
    }
    if (s == 4) {
        displayParserLog("Parsed a 2-qubit ParamDefStmt");
        defStmt->nqubits = 2;
        return defStmt;
    }
    if (s == 8) {
        displayParserLog("Parsed a 3-qubit ParamDefStmt");
        defStmt->nqubits = 3;
        return defStmt;
    }
    if (s == 16) {
        displayParserLog("Parsed a 4-qubit ParamDefStmt");
        defStmt->nqubits = 4;
        return defStmt;
    }
    displayParserWarning("Unsupported matrix size " + std::to_string(s));
    return defStmt;
}


std::unique_ptr<Statement> Parser::parseStatement_() {
    while (true) {
        if (curToken.type == TokenTy::Eof) {
            return nullptr;
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

    displayParserLog("ready to parse a statement");
    if (curToken.type == TokenTy::Circuit)
        return parseCircuitStmt_();
    if (curToken.type == TokenTy::Hash)
        return parseParameterDefStmt_();
    std::cerr << RED_FG << "curToken: " << curToken << RESET << "\n";
    displayParserError("Unknown token when trying to parse a statement");
    return nullptr;
}

std::unique_ptr<RootNode> Parser::parse() {
    auto root = std::make_unique<RootNode>();
    std::unique_ptr<Statement> stmt = nullptr;
    while ((stmt = parseStatement_()) != nullptr) {
        root->stmts.push_back(std::move(stmt));
    }
    if (curToken.type != TokenTy::Eof)
        displayParserWarning("end of parsing, but curToken is not EoF?");
    else
        displayParserLog("Reached EoF");
    return root;
}