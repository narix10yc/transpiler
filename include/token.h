#ifndef TOKEN_H_
#define TOKEN_H_

#include <string>

namespace openqasm {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int { 
    Eof = -1,
    Identifier = -2,
    Numeric = -3,

    // keywords
    If = -10,
    Then = -11,
    Else = -12,
    Openqasm = -13,
    Qreg = -14,
    Creg = -15,
    Gate = -16,

    // operators
    Add = -30,
    Sub = -31,
    Mul = -32,
    Div = -33,
    Greater = -34,
    Less = -35,
    GreaterEqual = -36,
    LessEqual = -37,

    Comma = -104,
    Semicolon = -105,
    L_RoundBraket = -106,
    R_RoundBraket = -107,
    L_SquareBraket = -108,
    R_SquareBraket = -109,
    L_AngleBraket = -110,
    R_AngleBraket = -111,
    L_CurlyBraket = -112,
    R_CurlyBraket = -113,

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
    Add, Sub, Mul, Div,
    None,
};


class Token {
public:
    TokenTy type;
    std::string str;
    Token(TokenTy type) : type(type), str("") {}
    Token(TokenTy type, const std::string& str) : type(type), str(str) {}

    std::string toString() const;

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
            default:
                return BinaryOp::None;
        }
    }
};

static std::string tokenTypetoString(TokenTy ty) {
    switch (ty) {
    case TokenTy::Eof: return "EOF";
    case TokenTy::Identifier: return "identifier";
    case TokenTy::Numeric: return "numeric";
    case TokenTy::If: return "if";
    case TokenTy::Else: return "else";
    case TokenTy::Then: return "then";
    // TODO
    default:
        return "<unknown>";
    }
}

static int getBinopPrecedence(BinaryOp op) {
    switch (op) {
        case BinaryOp::Greater: return 10;
        case BinaryOp::Less: return 10;
        case BinaryOp::GreaterEqual: return 10;
        case BinaryOp::LessEqual: return 10;
        case BinaryOp::Add: return 20;
        case BinaryOp::Sub: return 20;    
        case BinaryOp::Mul: return 40;
        case BinaryOp::Div: return 40;
        default:
            std::cerr << "Undefined binop precedence?\n";
            return -1;
    }
}

} // namespace openqasm

#endif // TOKEN_H_