#ifndef QUENCH_PARSER_H
#define QUENCH_PARSER_H

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <exception>
#include <queue>

#include "quench/ast.h"

namespace quench::ast {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int { 
    Eof = -1,
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
};

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
    int line;
    int column;
    std::string currentLine;
    std::ifstream file;

    Token curToken;
    Token nextToken;

    void displayParserError(const std::string& msg) const {
        // foreground color
        const std::string RED_FG = "\033[31m";
        const std::string GREEN_FG = "\033[32m";
        const std::string DEFAULT_FG = "\033[39m";
        const std::string RESET = "\033[0m";
        const std::string BOLD = "\033[1m";
        std::cerr << RED_FG << BOLD << "parser error: " << DEFAULT_FG << msg
                  << RESET << "\n"
                  << std::setw(5) << std::setfill(' ') << line << " | "
                  << currentLine << "\n"
                  << "      | " << std::string(static_cast<size_t>(column), ' ')
                  << GREEN_FG << BOLD << "^\n" << RESET;
    }

    int nextChar();
    int peekChar();

    /// @brief Proceed to the next Token. Update curToken and nextToken. Return
    /// false if EoF is reached.
    bool proceed();

    void skipRestOfLine();

    std::unique_ptr<GateApplyStmt> parseGateApplyStmt_();
    std::unique_ptr<CircuitStmt> parseCircuitStmt_();
public:
    Parser(const std::string& fileName)
        : line(0), column(0), currentLine(""), file(fileName),
          curToken(), nextToken() {}

    std::unique_ptr<RootNode> parse();
};




} // namespace quench::parse

#endif // QUENCH_PARSER_H