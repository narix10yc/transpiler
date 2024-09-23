#include "quench/parser.h"
#include <cassert>

using namespace quench;
using namespace quench::ast;
using namespace quench::quantum_gate;

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
        auto token = parseToken(col);
        col = token.colEnd;
        if (token.type != TokenTy::EndOfLine)
            tokenVec.push_back(token);
    }
    tokenIt = tokenVec.cbegin();

    std::cerr << Color::CYAN_FG << lineNumber << " | " << currentLine << "\n";
    // for (auto it = tokenVec.cbegin(); it != tokenVec.cend(); it++)
        // std::cerr << "col " << it->colStart << "-" << it->colEnd << "  " << (*it) << "\n";
    std::cerr << Color::RESET;

    return tokenVec.size();
}

Token Parser::parseToken(int col) {
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
        throwParserError("Unknown char " + std::to_string((int)(c)));
        assert(false && "Unknown char");
        return Token(TokenTy::Unknown, "", colStart, colStart+1);
    }
}

RootNode* Parser::parse() {
    readLine();
    auto root = new RootNode();
    while (true) {
        if (tokenIt->type == TokenTy::Circuit) {
            root->circuit = _parseCircuit();
            continue;
        }
        if (tokenIt->type == TokenTy::Hash) {
            auto defStmt = _parseParameterDefStmt();
            defStmt.gateMatrix.updateNqubits();
            root->paramDefs.push_back(defStmt);
            displayParserLog("Parsed param def #" + std::to_string(defStmt.refNumber));
            continue;
        }
        break;
    }
    return root;
}

QuantumCircuit Parser::_parseCircuit() {
    displayParserLog("ready to parse circuit");
    QuantumCircuit circuit;
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
                    throwParserError("Unsupported circuit argument '" + tokenIt->str +
                                     "' (expect either 'nqubits' or 'nparams')");
                proceedWithType(TokenTy::Equal);
                proceedWithType(TokenTy::Numeric);
                int num = convertCurTokenToInt();
                optionalProceedWithType(TokenTy::Comma);
                if (setFlag == 1) {
                    if (setNqubitsFlag >= 0)
                        displayParserWarning("Overwrite nqubits from " + std::to_string(setNqubitsFlag)
                                             + " to " + std::to_string(num));
                    circuit.nqubits = num;
                    setNqubitsFlag = num;
                    displayParserLog("nqubits updated to " + std::to_string(num));
                }
                else if (setFlag == 2) {
                    if (setNparamsFlag >= 0)
                        displayParserWarning("Overwrite nparams from " + std::to_string(setNparamsFlag)
                                             + " to " + std::to_string(num));
                    circuit.nparams = num;
                    setNparamsFlag = num;
                    displayParserLog("nparams updated to " + std::to_string(num));
                }
                continue;
            }

            throwParserError("Unexpected token type " + TokenTyToString(tokenIt->type) + " when expecting an Identifier");
        }
    }

    proceedWithType(TokenTy::Identifier);
    circuit.name = tokenIt->str;
    proceedWithType(TokenTy::L_CurlyBraket, true);
    displayParserLog("Ready to parse circuit " + circuit.name);

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
    return circuit;
}

GateParameter Parser::_parseGateParameter() {
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        int i = convertCurTokenToInt();
        proceed();
        return GateParameter(i);
    }
    assert(tokenIt->type == TokenTy::Numeric);
    double n = convertCurTokenToFloat();
    proceed();
    return GateParameter(n);
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
        else {
            proceed(); // eat '('
            while (true) {
                gate.params.push_back(_parseGateParameter());
                if (tokenIt->type == TokenTy::Comma) {
                    proceed();
                    continue;
                }
                if (tokenIt->type == TokenTy::Numeric || tokenIt->type == TokenTy::Percent)
                    continue;
                if (tokenIt->type == TokenTy::R_RoundBraket)
                    break;
                throwParserError("Unexpected token " + TokenTyToString(tokenIt->type));
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

ParameterDefStmt Parser::_parseParameterDefStmt() {
    assert(tokenIt->type == TokenTy::Hash);

    proceedWithType(TokenTy::Numeric);
    ParameterDefStmt defStmt(convertCurTokenToInt());
    displayParserLog("Ready to parse ParameterDef #" + std::to_string(defStmt.refNumber));

    proceedWithType(TokenTy::Equal);
    proceedWithType(TokenTy::L_CurlyBraket);
    proceed();

    quench::quantum_gate::matrix_t::p_matrix_t polyMatrix;
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
    defStmt.gateMatrix.matrix = std::move(polyMatrix);
    return defStmt;
}

std::complex<double> Parser::_parseComplexNumber() {
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
        throwParserError("Expect purely imaginary number to end with 'i'");
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
        throwParserError("Expect complex number to end with 'i'");

    proceed();
    return { real, imag };
}

saot::Polynomial Parser::_parseSaotPolynomial() {
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
            throwParserError("Expect ')'");
        }
        auto cplx = _parseComplexNumber();
        if (cplx.real() != 0.0 && cplx.imag() != 0.0) {
            displayParserWarning("Expect complex number to be enclosed by '()'");
        }
        return cplx;
    };

    const auto parseSaotVarSum = [&]() -> saot::VariableSumNode {
        saot::VariableSumNode N;
        // 'cos' 'sin'
        if (tokenIt->type != TokenTy::Identifier)
            throwParserError("Expect VariableSumNode to start with 'cos' or 'sin");
        if (tokenIt->str == "cos")
            N.op = saot::VariableSumNode::CosOp;
        else if (tokenIt->str == "sin")
            N.op = saot::VariableSumNode::SinOp;
        else
            throwParserError("Expect VariableSumNode to start with 'cos' or 'sin");
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
                throwParserError("Expect ')'");
        }
        else {
            if (nAdd > 0)
                displayParserWarning("Expect multiple mul terms to be enclosed by '()'");
        }
        return N;
    };

    const auto parseSaotMonomial = [&]() -> saot::Monomial {
        saot::Monomial M;
        M.coef = parseCoef();
        if (tokenIt->type == TokenTy::Mul) {
            if (M.coef == std::complex<double>(1.0, 0.0))
                displayParserWarning("Top-level '*'");
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
                throwParserError("Expect '%' in parsing ExpiVars");
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
                throwParserError("Expect ')'");
        }
        else {
            if (M.expiVars().size() > 1)
                displayParserWarning("Expect multiple expi terms to be enclosed by '()'");
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
