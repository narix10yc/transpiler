#include "saot/Parser.h"
#include "saot/CircuitGraph.h"
#include "saot/ast.h"

#include "utils/iocolor.h"

using namespace saot;
using namespace saot::ast;
using namespace IOColor;

std::string saot::getNameOfTokenKind(TokenKind kind) {
  switch (kind) {
  case tk_Eof:
    return "EoF";
  case tk_LineFeed:
    return "\\n";
  case tk_Numeric:
    return "Num";
  case tk_Identifier:
    return "Identifier";

  case tk_L_RoundBracket:
    return "(";
  case tk_R_RoundBracket:
    return ")";
  case tk_L_CurlyBracket:
    return "{";
  case tk_R_CurlyBracket:
    return "}";

  case tk_Comma:
    return ",";
  case tk_Semicolon:
    return ";";

  default:
    return "Unimplemented Name of TokenKind " +
           std::to_string(static_cast<int>(kind));
  }
}

void Parser::printLocation(std::ostream& os) const {
  auto lineInfo = lexer.getCurLineInfo();
  os << std::setw(5) << std::setfill(' ') << lineInfo.line << " | ";
  os.write(lineInfo.memRefBegin, lineInfo.memRefEnd - lineInfo.memRefBegin);
  os << "      | "
     << std::string(curToken.memRefBegin - lexer.lineBegin,' ')
     << BOLDGREEN(std::string(curToken.length(), '^') << "\n");
}

std::ostream& Token::print(std::ostream& os) const {
  os << "tok(";

  if (kind == tk_Numeric) {
    assert(memRefBegin != memRefEnd);
    os << "Num,";
    return os.write(memRefBegin, memRefEnd - memRefBegin) << ")";
  }

  if (kind == tk_Identifier) {
    assert(memRefBegin != memRefEnd);
    os << "Identifier,";
    return os.write(memRefBegin, memRefEnd - memRefBegin) << ")";
  }

  os << IOColor::CYAN_FG;
  switch (kind) {
  case tk_Unknown:
    os << "Unknown";
    break;
  case tk_Eof:
    os << "EoF";
    break;
  case tk_LineFeed:
    os << "\\n";
    break;

  case tk_L_RoundBracket:
    os << "(";
    break;
  case tk_R_RoundBracket:
    os << ")";
    break;
  case tk_L_SquareBracket:
    os << "[";
    break;
  case tk_R_SquareBracket:
    os << "]";
    break;
  case tk_L_CurlyBracket:
    os << "{";
    break;
  case tk_R_CurlyBracket:
    os << "}";
    break;
  case tk_Less:
    os << "<";
    break;
  case tk_Greater:
    os << ">";
    break;

  case tk_Comma:
    os << ",";
    break;
  case tk_Semicolon:
    os << ";";
    break;
  case tk_Percent:
    os << "%";
    break;
  case tk_AtSymbol:
    os << "@";
    break;

  default:
    os << static_cast<int>(kind) << " Not Imp'ed";
    break;
  }

  return os << IOColor::RESET << ")";
}

void Lexer::lex(Token& tok) {
  const auto assignTok = [&](TokenKind kind, int nchars = 1) {
    tok = Token(kind, curPtr, curPtr + nchars);
  };

  if (curPtr >= bufferEnd) {
    assignTok(tk_Eof);
    return;
  }

  char c = *curPtr++;
  while (c == ' ' || c == '\r')
    c = *curPtr++;

  switch (c) {
  case '\0':
    assignTok(tk_Eof);
    return;
  case '\n': {
    assignTok(tk_LineFeed);
    ++line;
    lineBegin = curPtr;
    return;
  }

  case '(':
    assignTok(tk_L_RoundBracket);
    return;
  case ')':
    assignTok(tk_R_RoundBracket);
    return;
  case '[':
    assignTok(tk_L_SquareBracket);
    return;
  case ']':
    assignTok(tk_R_SquareBracket);
    return;
  case '{':
    assignTok(tk_L_CurlyBracket);
    return;
  case '}':
    assignTok(tk_R_CurlyBracket);
    return;
  case '<':
    assignTok(tk_Less);
    return;
  case '>':
    assignTok(tk_Greater);
    return;

  case ',':
    assignTok(tk_Comma);
    return;
  case ';':
    assignTok(tk_Semicolon);
    return;
  case '%':
    assignTok(tk_Percent);
    return;
  case '@':
    assignTok(tk_AtSymbol);
    return;

  // '*' or '**'
  case '*': {
    if (*curPtr++ == '*')
      assignTok(tk_Pow, 2);
    else {
      --curPtr;
      assignTok(tk_Mul);
    }
    return;
  }

  default:
    auto* memRefBegin = --curPtr;
    if (c == '-' || ('0' <= c && c <= '9')) {
      c = *(++curPtr);
      while (c == 'e' || c == '+' || c == '-' || c == '.' ||
             ('0' <= c && c <= '9'))
        c = *(++curPtr);
      tok = Token(tk_Numeric, memRefBegin, curPtr);
      return;
    }
    assert(std::isalpha(c) && "Can only parse identifiers now");
    c = *(++curPtr);
    while (c == '_' || std::isalnum(c))
      c = *(++curPtr);
    tok = Token(tk_Identifier, memRefBegin, curPtr);
    return;
  }
}

void Lexer::skipLine() {
  while (curPtr < bufferEnd) {
    if (*curPtr++ == '\n') {
      ++line;
      lineBegin = curPtr;
      break;
    }
  }
}

Lexer::LineInfo Lexer::getCurLineInfo() const {
  auto* lineEnd = curPtr;
  while (lineEnd < bufferEnd) {
    if (*lineEnd++ == '\n')
      break;
  }
  return LineInfo{
      .line = line, .memRefBegin = lineBegin, .memRefEnd = lineEnd};
}

std::complex<double> Parser::parseComplexNumber() {
  double multiple = 1.0;
  // general complex number, parenthesis required
  if (optionalAdvance(tk_L_RoundBracket)) {
    if (curToken.is(tk_Sub)) {
      advance(tk_Sub);
      multiple = -1.0;
    }
    requireCurTokenIs(tk_Numeric);
    double re = multiple * curToken.toDouble();
    advance(tk_Numeric);

    if (curToken.is(tk_Add))
      multiple = 1.0;
    else if (curToken.is(tk_Sub))
      multiple = -1.0;
    else
      logErr() << "Expect '+' or '-' when parsing a general complex number";
    advance();

    requireCurTokenIs(tk_Numeric);
    double im = multiple * curToken.toDouble();
    advance(tk_Numeric);

    requiredAdvance(tk_R_RoundBracket);
    return {re, im};
  }

  multiple = 1.0;
  if (curToken.is(tk_Sub)) {
    advance(tk_Sub);
    multiple = -1.0;
  }

  // i or -i
  if (curToken.isI()) {
    advance(tk_Identifier);
    return {0.0, multiple};
  }

  // purely real or purely imaginary
  if (curToken.is(tk_Numeric)) {
    double value = multiple * curToken.toDouble();
    advance(tk_Numeric);
    if (curToken.isI()) {
      advance(tk_Identifier);
      return {0.0, value};
    }
    return {value, 0.0};
  }

  logErr() << "Unable to parse complex number\n";
  return 0.0;
}

GateApplyStmt Parser::parseGateApply() {
  requireCurTokenIs(tk_Identifier);
  GateApplyStmt stmt(std::string(curToken.memRefBegin, curToken.memRefEnd), {});
  advance(tk_Identifier);

  // parameters (optional)
  if (optionalAdvance(tk_L_RoundBracket)) {
    if (optionalAdvance(tk_Hash)) {
      // parameter ref #N
      requireCurTokenIs(tk_Numeric);
      stmt.argument = curToken.toInt();
      advance(tk_Numeric);
    } else {
      // up to 3 parameters
      stmt.argument.set<GateMatrix::gate_params_t>();
      auto& parameters = stmt.argument.get<GateMatrix::gate_params_t>();
      for (int i = 0; i < 3; ++i) {
        if (curToken.is(tk_Numeric)) {
          parameters[i] = curToken.toDouble();
        } else if (optionalAdvance(tk_Percent)) {
          requireCurTokenIs(tk_Numeric);
          parameters[i] = curToken.toInt();
        } else {
          for (int j = i; j < 3; ++j)
            parameters[j].reset();
          break;
        }
        advance();
        optionalAdvance(tk_Comma);
      }
    }
    requiredAdvance(tk_R_RoundBracket);
  }

  // target qubits
  while (true) {
    if (curToken.is(tk_Numeric)) {
      stmt.qubits.push_back(curToken.toInt());
      advance(tk_Numeric);
      optionalAdvance(tk_Comma);
      continue;
    }
    break;
  }

  // std::cerr << "Parsed gate " << stmt.name << " with "
  //           << stmt.qubits.size() << " targets, "
  //           << "curToken is " << getNameOfTokenKind(curToken.kind)
  //           << "\n";
  return stmt;
}

GateChainStmt Parser::parseGateChain() {
  GateChainStmt stmt;
  while (true) {
    stmt.gates.push_back(parseGateApply());
    if (optionalAdvance(tk_Semicolon))
      return stmt;
    if (optionalAdvance(tk_AtSymbol))
      continue;
    logErr() << "Unexpected token '" << getNameOfTokenKind(curToken.kind)
             << "' Did you forget ';'?\n";
    return stmt;
  }
}

QuantumCircuit Parser::parseQuantumCircuit() {
  requireCurTokenIs(tk_Identifier);
  if (std::string(curToken.memRefBegin, curToken.memRefEnd) != "circuit") {
    printLocation(
        logErr() << "Expect 'circuit' keyword in parsing QuantumCircuit\n");
    failAndExit();
  }
  advance(tk_Identifier);

  requireCurTokenIs(tk_Identifier, "Expecting a name");
  QuantumCircuit circuit(std::string(curToken.memRefBegin, curToken.memRefEnd));
  advance(tk_Identifier);
  skipLineBreaks();

  requiredAdvance(tk_L_CurlyBracket);
  skipLineBreaks();

  // circuit body
  while (true) {
    if (optionalAdvance(tk_R_CurlyBracket))
      break;
    circuit.addChainStmt(std::make_unique<GateChainStmt>(parseGateChain()));
    skipLineBreaks();
  }

  return circuit;
}
