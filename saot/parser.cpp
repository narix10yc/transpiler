#include "saot/Parser.h"
#include <cassert>

#include "utils/iocolor.h"
using namespace IOColor;

using namespace saot;
using namespace saot::ast;

void LegacyParser::throwLegacyParserError(const char* msg, std::ostream& os) const {
    os << RED_FG << BOLD << "LegacyParser error: "
       << DEFAULT_FG << msg << RESET << "\n"
       << std::setw(5) << std::setfill(' ') << lineNumber << " | "
       << currentLine << "\n"
       << "      | " << std::string(static_cast<size_t>(tokenIt->colStart), ' ')
       << GREEN_FG << BOLD
       << std::string(static_cast<size_t>(tokenIt->colEnd - tokenIt->colStart), '^')
       << "\n" << RESET;
    throw std::runtime_error("LegacyParser error");
}

std::ostream& LegacyParser::displayLegacyParserWarning(const char* msg, std::ostream& os) const {
    return os << YELLOW_FG << BOLD << "LegacyParser warning: "
              << DEFAULT_FG << msg << RESET << "\n"
              << std::setw(5) << std::setfill(' ') << lineNumber << " | "
              << currentLine << "\n"
              << "      | " << std::string(static_cast<size_t>(tokenIt->colStart), ' ')
              << GREEN_FG << BOLD
              << std::string(static_cast<size_t>(tokenIt->colEnd - tokenIt->colStart), '^')
              << "\n" << RESET;
}

std::ostream& LegacyParser::displayLegacyParserLog(const char* msg, std::ostream& os) const {
    return os << CYAN_FG << BOLD << "LegacyParser log: "
              << DEFAULT_FG << msg << RESET << "\n";
}

int LegacyParser::readLine() {
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
        auto token = parseToken(col);
        col = token.colEnd;
        if (token.type != TokenTy::EndOfLine)
            tokenVec.push_back(token);
    }
    tokenIt = tokenVec.cbegin();

    std::cerr << CYAN_FG << lineNumber << " | " << currentLine << "\n";
    // for (auto it = tokenVec.cbegin(); it != tokenVec.cend(); it++)
        // std::cerr << "col " << it->colStart << "-" << it->colEnd << "  " << (*it) << "\n";
    std::cerr << RESET;

    return tokenVec.size();
}

Token LegacyParser::parseToken(int col) {
    if (col >= lineLength)
        return Token(TokenTy::EndOfLine, "", lineLength, lineLength+1);

    int curCol = col;
    char c = currentLine[curCol];
    while (c == ' ')
        c = currentLine[++curCol];
    if (curCol >= lineLength)
        return Token(TokenTy::EndOfLine, "", lineLength, lineLength+1);
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
        throwLegacyParserError("Unknown char " + std::to_string((int)(c)));
        assert(false && "Unknown char");
        return Token(TokenTy::Unknown, "", colStart, colStart+1);
    }
}

QuantumCircuit LegacyParser::parse() {
    readLine();
    QuantumCircuit qc;
    while (true) {
        if (tokenIt->type == TokenTy::Circuit) {
            _parseCircuitBody(qc);
            continue;
        }
        if (tokenIt->type == TokenTy::Hash) {
            auto defStmt = _parseParameterDefStmt();
            qc.paramDefs.push_back(defStmt);
            displayLegacyParserLog("Parsed param def #" + std::to_string(defStmt.refNumber));
            continue;
        }
        break;
    }
    return qc;
}

QuantumCircuit& LegacyParser::_parseCircuitBody(QuantumCircuit& qc) {
    displayLegacyParserLog("ready to parse circuit");
    int setNqubitsFlag = -1;
    int setNparamsFlag = -1;
    int setFlag = 0;
    
    // optional <n qubits, p parameters>
    if (optionalProceedWithType(TokenTy::Less)) {
        while (true) {
            proceed();
            setFlag = 0;
            if (tokenIt->type == TokenTy::Greater) {
                // proceed(); // eat '>'
                break;
            }
            if (tokenIt->type == TokenTy::Identifier) {
                if (tokenIt->str == "nqubits")
                    setFlag = 1;
                else if (tokenIt->str == "nparams")
                    setFlag = 2;
                else
                    throwLegacyParserError("Unsupported circuit argument '" + tokenIt->str +
                                     "' (expect either 'nqubits' or 'nparams')");
                proceedWithType(TokenTy::Equal);
                proceedWithType(TokenTy::Numeric);
                int num = convertCurTokenToInt();
                optionalProceedWithType(TokenTy::Comma);
                if (setFlag == 1) {
                    if (setNqubitsFlag >= 0)
                        displayLegacyParserWarning("Overwrite nqubits from " + std::to_string(setNqubitsFlag)
                                             + " to " + std::to_string(num));
                    qc.nqubits = num;
                    setNqubitsFlag = num;
                    displayLegacyParserLog("nqubits updated to " + std::to_string(num));
                }
                else if (setFlag == 2) {
                    if (setNparamsFlag >= 0)
                        displayLegacyParserWarning("Overwrite nparams from " + std::to_string(setNparamsFlag)
                                             + " to " + std::to_string(num));
                    qc.nparams = num;
                    setNparamsFlag = num;
                    displayLegacyParserLog("nparams updated to " + std::to_string(num));
                }
                continue;
            }

            throwLegacyParserError("Unexpected token type " + TokenTyToString(tokenIt->type) + " when expecting an Identifier");
        }
    }

    proceedWithType(TokenTy::Identifier);
    qc.name = tokenIt->str;
    proceedWithType(TokenTy::L_CurlyBraket, true);
    displayLegacyParserLog("Ready to parse circuit " + qc.name);

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
                throwLegacyParserError("Unexpected token type " + TokenTyToString(tokenIt->type)
                                + " when expecting either AtSymbol or Semicolon");
            }
            qc.stmts.push_back(std::make_unique<GateChainStmt>(chain));
            continue;
        }
        break;
    } 
    
    if (tokenIt->type != TokenTy::R_CurlyBraket) {
        throwLegacyParserError("Unexpected token " + tokenIt->to_string());
    }
    proceed(); // eat '}'
    displayLegacyParserLog("Parsed a circuit with " + std::to_string(qc.stmts.size()) + " chains");
    return qc;
}

GateMatrix::gate_params_t LegacyParser::_parseParams_t() {
    GateMatrix::gate_params_t p;
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        p[0] = convertCurTokenToInt();
        proceed();
    }
    else if (tokenIt->type == TokenTy::Numeric || tokenIt->type == TokenTy::Sub)
        p[0] = _LegacyParserealNumber();
    else
        return p;

    optionalProceedWithType(TokenTy::Comma);
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        p[0] = convertCurTokenToInt();
        proceed();
    }
    else if (tokenIt->type == TokenTy::Numeric || tokenIt->type == TokenTy::Sub)
        p[0] = _LegacyParserealNumber();
    else
        return p;

    optionalProceedWithType(TokenTy::Comma);
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        p[0] = convertCurTokenToInt();
        proceed();
    }
    else if (tokenIt->type == TokenTy::Numeric || tokenIt->type == TokenTy::Sub)
        p[0] = _LegacyParserealNumber();
    else
        return p;
    return p;
}

static int getNumActiveParams(const GateMatrix::gate_params_t& params) {
    for (int i = 0; i < params.size(); i++) {
        if (params[i].index() == 0)
            return i;
    }
    return params.size();
}

GateApplyStmt LegacyParser::_parseGateApply() {
    assert(tokenIt->type == TokenTy::Identifier);
    GateApplyStmt gate(tokenIt->str);

    if (optionalProceedWithType(TokenTy::L_RoundBraket)) {
        proceed();
        if (tokenIt->type == TokenTy::Hash) {
            proceedWithType(TokenTy::Numeric);
            gate.paramRefOrMatrix = convertCurTokenToInt();
        }
        else {
            gate.paramRefOrMatrix = _parseParams_t();
            if (tokenIt->type != TokenTy::R_RoundBraket)
                throwLegacyParserError("Expect ')' to end gate parameter");
            displayLegacyParserLog("Parsed " +
                std::to_string(getNumActiveParams(std::get<GateMatrix::gate_params_t>(gate.paramRefOrMatrix))) + " parameters");
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
        throwLegacyParserError("Gate " + gate.name + " has no target");
    
    displayLegacyParserLog("Parsed gate " + gate.name + " with " +
                     std::to_string(gate.qubits.size()) + " targets");
    return gate;
}

ParameterDefStmt LegacyParser::_parseParameterDefStmt() {
    assert(tokenIt->type == TokenTy::Hash);

    proceedWithType(TokenTy::Numeric);
    ParameterDefStmt defStmt(convertCurTokenToInt());
    displayLegacyParserLog("Ready to parse ParameterDef #" + std::to_string(defStmt.refNumber));

    proceedWithType(TokenTy::Equal);
    proceedWithType(TokenTy::L_CurlyBraket);
    proceed();

    GateMatrix::p_matrix_t polyMatrix;
    while (true) {
        auto poly = _parseSaotPolynomial();
        poly.print(std::cerr) << "\n";
        polyMatrix.data.push_back(poly);
        if (tokenIt->type == TokenTy::Comma) {
            proceed();
        }
        if (tokenIt->type == TokenTy::R_CurlyBraket) {
            proceed();
            break;
        }
    }
    polyMatrix.updateSize();
    defStmt.gateMatrix._matrix = std::move(polyMatrix);
    return defStmt;
}

double LegacyParser::_LegacyParserealNumber() {
    double m = 1.0;
    while (true) {
        if (tokenIt->type == TokenTy::Sub) {
            proceed();
            m *= -1.0;
            continue;
        }
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    if (tokenIt->type == TokenTy::Numeric) {
        double num = m * convertCurTokenToFloat();
        proceed();
        return num;
    }
    throwLegacyParserError("Unknown token when parsing real number");
    return 0.0;
}

std::complex<double> LegacyParser::_parseComplexNumber() {
    double m = 1.0;
    while (true) {
        if (tokenIt->type == TokenTy::Sub) {
            proceed();
            m *= -1.0;
            continue;
        }
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    double real = m;
    if (tokenIt->type == TokenTy::Numeric) {
        real *= convertCurTokenToFloat();
        proceed();
    }
    else if (tokenIt->type == TokenTy::Identifier) {
        if (tokenIt->str == "i") {
            proceed();
            return { 0.0, real };
        }
        throwLegacyParserError("Expect purely imaginary number to end with 'i'");
    }
    // just one part (pure real or pure imag)
    if (tokenIt->type != TokenTy::Add && tokenIt->type != TokenTy::Sub)
        return { real, 0.0 };

    m = 1.0;
    while (true) {
        if (tokenIt->type == TokenTy::Sub) {
            proceed();
            m *= -1.0;
            continue;
        }
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    double imag = m * convertCurTokenToFloat();
    proceed();
    if (!(tokenIt->type == TokenTy::Identifier && tokenIt->str == "i"))
        throwLegacyParserError("Expect complex number to end with 'i'");

    proceed();
    return { real, imag };
}

saot::Polynomial LegacyParser::_parseSaotPolynomial() {
    const auto parseCoef = [&]() -> std::complex<double> {
        if (tokenIt->type == TokenTy::Identifier) {
            if (tokenIt->str == "i") {
                proceed();
                return { 0.0, 1.0 };
            }
            return { 1.0, 0.0 };
        }
        bool paranFlag = false;
        if (tokenIt->type == TokenTy::L_RoundBraket) {
            paranFlag = true;
            proceed();
            auto cplx = _parseComplexNumber();
            if (tokenIt->type == TokenTy::R_RoundBraket) {
                proceed();
                return cplx;
            }
            throwLegacyParserError("Expect ')'");
        }
        auto cplx = _parseComplexNumber();
        if (cplx.real() != 0.0 && cplx.imag() != 0.0) {
            displayLegacyParserWarning("Expect complex number to be enclosed by '()'");
        }
        return cplx;
    };

    const auto parseSaotVarSum = [&]() -> saot::VariableSumNode {
        saot::VariableSumNode N;
        // 'cos' 'sin'
        if (tokenIt->type != TokenTy::Identifier)
            throwLegacyParserError("Expect VariableSumNode to start with 'cos' or 'sin");
        if (tokenIt->str == "cos")
            N.op = saot::VariableSumNode::CosOp;
        else if (tokenIt->str == "sin")
            N.op = saot::VariableSumNode::SinOp;
        else
            throwLegacyParserError("Expect VariableSumNode to start with 'cos' or 'sin");
        proceed();

        // optional '('
        bool paranFlag = false;
        if (tokenIt->type == TokenTy::L_RoundBraket) {
            paranFlag = true;
            proceed();
        }

        // terms
        int nAdd = 0;
        while (true) {
            if (tokenIt->type == TokenTy::Percent) {
                proceedWithType(TokenTy::Numeric);
                N.addVar(convertCurTokenToInt());
                proceed();
            }
            else if (tokenIt->type == TokenTy::Numeric) {
                N.constant = convertCurTokenToFloat();
                proceed();
                break;
            }
            if (tokenIt->type == TokenTy::Add) {
                nAdd++;
                proceed();
                continue;
            }
            break;
        }

        // matching (optional) ')'
        if (paranFlag) {
            if (tokenIt->type == TokenTy::R_RoundBraket)
                proceed();
            else
                throwLegacyParserError("Expect ')'");
        }
        else {
            if (nAdd > 0)
                displayLegacyParserWarning("Expect multiple mul terms to be enclosed by '()'");
        }
        return N;
    };

    const auto parseSaotMonomial = [&]() -> saot::Monomial {
        saot::Monomial M;
        M.coef = parseCoef();
        if (tokenIt->type == TokenTy::Mul) {
            if (M.coef == std::complex<double>(1.0, 0.0))
                displayLegacyParserWarning("Top-level '*'");
            proceed();
        }

        // mul term
        while (true) {
            if (tokenIt->type == TokenTy::Identifier && tokenIt->str == "expi")
               break;
            M.insertMulTerm(parseSaotVarSum());
            if (tokenIt->type == TokenTy::Mul) {
                proceed();
                continue;
            }
            break;
        }

        // expi term
        if (tokenIt->type != TokenTy::Identifier || tokenIt->str != "expi")
            return M;
        
        proceed();
        bool paranFlag = false;
        if (tokenIt->type == TokenTy::L_RoundBraket) {
            paranFlag = true;
            proceed();
        }
        while (true) {
            if (tokenIt->type != TokenTy::Percent)
                throwLegacyParserError("Expect '%' in parsing ExpiVars");
            proceedWithType(TokenTy::Numeric);
            M.insertExpiVar(convertCurTokenToInt());
            if (optionalProceedWithType(TokenTy::Add)) {
                proceed();
                continue;
            }
            proceed();
            break;
        }
        if (paranFlag) {
            if (tokenIt->type == TokenTy::R_RoundBraket)
                proceed();
            else
                throwLegacyParserError("Expect ')'");
        }
        else {
            if (M.expiVars().size() > 1)
                displayLegacyParserWarning("Expect multiple expi terms to be enclosed by '()'");
        }
        return M;
    };

    saot::Polynomial P;
    while (true) {
        P.insertMonomial(parseSaotMonomial());
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }

    return P;
}
