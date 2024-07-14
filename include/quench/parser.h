#ifndef QUENCH_PARSER_H
#define QUENCH_PARSER_H

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <exception>
#include <queue>
#include <sstream>

#include "quench/ast.h"

namespace quench::ast {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int { 
    Eof = -1,
    Start = -4,
    Identifier = -2,
    Numeric = -3,

    // keywords
    Circuit = -10,

    // operators
    Add = -30,                  // +
    Sub = -31,                  // -
    Mul = -32,                  // *
    Div = -33,                  // /
    Pow = -34,                  // **
    Greater = -35,              // >
    Less = -36,                 // <
    Equal = -37,                // =
    GreaterEqual = -38,         // >=
    LessEqual = -39,            // <=
    EqualEqual = -40,           // ==

    // symbols
    Comma = -104,               // ,
    Semicolon = -105,           // ;
    L_RoundBraket = -106,       // (
    R_RoundBraket = -107,       // )
    L_SquareBraket = -108,      // [
    R_SquareBraket = -109,      // ]
    L_CurlyBraket = -112,       // {
    R_CurlyBraket = -113,       // }
    SingleQuote = -114,         // '
    DoubleQuote = -115,         // "
    AtSymbol = -116,            // @
    Percent = -117,             // %
    Hash = -118,                // #
    Backslash = -119,           // '\'
    Comment = -120,             // '//'
    CommentStart = -121,        // '/*'
    CommentEnd = -122,          // '*/'

    LineFeed = 10, // '\n'
    CarriageReturn = 13, // '\r'

    Unknown = -1000,
    Any = -1001,
};

static std::string TokenTyToString(TokenTy ty) {
    switch (ty) {
    case TokenTy::Eof: return "EoF";
    case TokenTy::Start: return "Start";
    case TokenTy::Identifier: return "Identifier";
    case TokenTy::Numeric: return "Numeric";
    
    case TokenTy::Circuit: return "Circuit";

    case TokenTy::Add: return "Add";
    case TokenTy::Sub: return "Sub";
    case TokenTy::Mul: return "Mul";
    case TokenTy::Div: return "Div";
    case TokenTy::Pow: return "Pow";

    case TokenTy::Comma: return "Comma";
    case TokenTy::Semicolon: return "Semicolon";
    case TokenTy::L_RoundBraket: return "L_RoundBraket";
    case TokenTy::R_RoundBraket: return "R_RoundBraket";
    case TokenTy::L_SquareBraket: return "L_SquareBraket";
    case TokenTy::R_SquareBraket: return "R_SquareBraket";
    case TokenTy::L_CurlyBraket: return "L_CurlyBraket";
    case TokenTy::R_CurlyBraket: return "R_CurlyBraket";
    case TokenTy::SingleQuote: return "SingleQuote";
    case TokenTy::DoubleQuote: return "DoubleQuote";
    case TokenTy::Hash: return "Hash";

    default: return "'Not Implemented TokenTy'";
    }
}

enum class UnaryOp {
    Positive, Negative, None,
};

enum class BinaryOp {
    Greater, Less,
    GreaterEqual, LessEqual,
    Add, Sub, Mul, Div, Pow, 
    None,
};

class Token {
public:
    TokenTy type;
    std::string str;

    Token() : type(TokenTy::Unknown), str() {}
    Token(TokenTy type, const std::string& str="") : type(type), str(str) {}
    Token(const std::string& str) : type(TokenTy::Identifier), str(str) {}

    BinaryOp toBinaryOp() const {
        switch (type) {
            case TokenTy::Greater: return BinaryOp::Greater;
            case TokenTy::Less: return BinaryOp::Less;
            case TokenTy::GreaterEqual: return BinaryOp::GreaterEqual;
            case TokenTy::LessEqual: return BinaryOp::LessEqual;
            case TokenTy::Add: return BinaryOp::Add;
            case TokenTy::Sub: return BinaryOp::Sub;    
            case TokenTy::Mul: return BinaryOp::Mul;
            case TokenTy::Div: return BinaryOp::Div;
            case TokenTy::Pow: return BinaryOp::Pow;
            default:
                return BinaryOp::None;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Token& token) {
        switch (token.type) {
            case TokenTy::Eof: return os << "EoF";
            case TokenTy::Start: return os << "Start";
            case TokenTy::Numeric: return os << "Num(" << token.str << ")";
            case TokenTy::Identifier: return os << "Identifier("
                                                << token.str << ")";

            case TokenTy::Circuit: return os << "'circuit'";

            case TokenTy::Add: return os << "'+'";
            case TokenTy::Sub: return os << "'-'";
            case TokenTy::Mul: return os << "'*'";
            case TokenTy::Div: return os << "'/'";
            case TokenTy::Equal: return os << "'='";
            case TokenTy::EqualEqual: return os << "'=='";
            case TokenTy::Less: return os << "'<'";
            case TokenTy::LessEqual: return os << "'<='";
            case TokenTy::Greater: return os << "'>'";
            case TokenTy::GreaterEqual: return os << "'>='";

            case TokenTy::Comma: return os << "','";
            case TokenTy::Semicolon: return os << "';'";
            case TokenTy::L_RoundBraket: return os << "'('";
            case TokenTy::R_RoundBraket: return os << "')'";
            case TokenTy::L_SquareBraket: return os << "'['";
            case TokenTy::R_SquareBraket: return os << "']'";
            case TokenTy::L_CurlyBraket: return os << "'{'";
            case TokenTy::R_CurlyBraket: return os << "'}'";
            case TokenTy::SingleQuote: return os << "'SingleQuote'";
            case TokenTy::DoubleQuote: return os << "'\"'";
            case TokenTy::Hash: return os << "'#'";
            case TokenTy::Comment: return os << "'Comment'";
            case TokenTy::CommentStart: return os << "'CommentStart'";
            case TokenTy::CommentEnd: return os << "'CommentEnd'";

            case TokenTy::LineFeed: return os << "'\\n'";
            case TokenTy::CarriageReturn: return os << "'\\r'";

            case TokenTy::Unknown: return os << "'Unknown'";
            default: return os << "'Not Implemented'";
        }
    }
};

class Parser {
protected:
    const std::string RED_FG = "\033[31m";
    const std::string YELLOW_FG = "\033[33m";
    const std::string GREEN_FG = "\033[32m";
    const std::string CYAN_FG = "\033[36m";
    const std::string DEFAULT_FG = "\033[39m";
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";

    int line;
    int column;
    std::string currentLine;
    std::ifstream file;

    Token curToken;
    Token nextToken;

    void throwParserError(const std::string& msg) const {
        std::cerr << RED_FG << BOLD << "parser error: " << DEFAULT_FG
                  << msg << RESET << "\n"
                  << std::setw(5) << std::setfill(' ') << line << " | "
                  << currentLine << "\n"
                  << "      | " << std::string(static_cast<size_t>(column), ' ')
                  << GREEN_FG << BOLD << "^\n" << RESET;
        throw std::runtime_error("parser error");
    }

    void displayParserWarning(const std::string& msg) const {
        std::cerr << YELLOW_FG << BOLD << "parser warning: "
                  << DEFAULT_FG << msg << RESET << "\n"
                  << std::setw(5) << std::setfill(' ') << line << " | "
                  << currentLine << "\n";

    }

    void displayParserLog(const std::string& msg) const {
        std::cerr << CYAN_FG << BOLD << "parser log: "
                << DEFAULT_FG << msg << RESET << "\n";
    }

    /// @brief read one line from file. Return the length of the read line.
    /// @return -1 if EoF is reached. In such case file.close() will be called.
    /// Otherwise, return the length of the read line.
    int readLine();

    int nextChar();
    int peekChar();

    /// @brief Proceed to the next Token. curToken is replaced by nextToken.
    /// nextToken is updated by reading the next (few) characters.
    /// return false if EoF is reached (i.e. curToken is EoF after procession)
    bool proceed();

    void skipLineFeeds() {
        while (curToken.type == TokenTy::LineFeed)
            proceed();
    }

    double convertCurTokenToFloat() const {
        assert(curToken.type == TokenTy::Numeric);
        int count = 0;
        for (const auto& c : curToken.str) {
            if (c == '.')
                count++;
        }
        if (count > 1)
            throwParserError("Unable to parse '" + curToken.str + "' to float");
        return std::stod(curToken.str);
    }

    int convertCurTokenToInt() const {
        assert(curToken.type == TokenTy::Numeric);
        int count = 0;
        for (const auto& c : curToken.str) {
            if (c == '.')
                count++;
        }
        if (count > 1 || (count == 1 && curToken.str.back() != '.'))
            throwParserError("Unable to parse '" + curToken.str + "' to int");
        return std::stod(curToken.str);
    }

    /// @brief Proceed to the next Token with the expectation that the next
    /// Token has a specific type. Procession takes place in case of match.
    /// @param ty Next Token's type
    /// @param must_match if set to true, display error in case of mismatch 
    /// @return bool: is the target type met? 
    bool proceedWithType(TokenTy ty, bool must_match=true) {
        if (nextToken.type == ty) {
            proceed();
            return true;
        }
        if (must_match) {
            std::stringstream ss;
            ss << "Expecting token type " << TokenTyToString(ty) << ", "
                << "but nextToken is " << nextToken;
            throwParserError(ss.str());
        }
        return false;
    }

    cas::Polynomial parsePolynomial_();

    GateApplyStmt parseGateApplyStmt_();
    GateBlockStmt parseGateBlockStmt_();
    CircuitStmt parseCircuitStmt_();

    ParameterDefStmt parseParameterDefStmt_();

    bool parseStatement_(RootNode&);
public:
    Parser(const std::string& fileName)
        : line(0), column(0), currentLine(""), file(fileName),
          curToken(TokenTy::Start), nextToken(TokenTy::Start) {}

    RootNode parse();
};




} // namespace quench::parse

#endif // QUENCH_PARSER_H