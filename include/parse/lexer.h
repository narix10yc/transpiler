#ifndef PARSE_LEXER_H
#define PARSE_LEXER_H

#include <string>
#include <iostream>
#include <fstream>

namespace parse {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int { 
    Eof = -1,
    Identifier = -2,
    Numeric = -3,

    // keywords
    Circuit = -10,
    Gate = -17,

    // operators
    Add = -30,                  // +
    Sub = -31,                  // -
    Mul = -32,                  // *
    Div = -33,                  // /
    Pow = -38,                  // **
    Greater = -34,              // >
    Less = -35,                 // <
    GreaterEqual = -36,         // >=
    LessEqual = -37,            // <=

    Comma = -104,               // ,
    Semicolon = -105,           // ;
    L_RoundBraket = -106,       // (
    R_RoundBraket = -107,       // )
    L_SquareBraket = -108,      // [
    R_SquareBraket = -109,      // ]
    L_AngleBraket = -110,       // <
    R_AngleBraket = -111,       // >
    L_CurlyBraket = -112,       // {
    R_CurlyBraket = -113,       // }
    SingleQuote = -114,         // '
    DoubleQuote = -115,         // "

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
    union {
      std::string str;
      double num;  
    };
    explicit Token(TokenTy type) : type(type), num(0) {}
    explicit Token(const std::string& str) : type(TokenTy::Identifier), str(str) {}
    explicit Token(double num) : type(TokenTy::Numeric), num(num) {}

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
};

class Lexer {
    int line;
    int column;
    std::string currentLine;
    std::string fileName;
public:
    Lexer(const std::string fileName)
        : line(0), column(0), currentLine(), fileName(fileName) {}
};




} // namespace parse

#endif // PARSE_LEXER_H