#include "openqasm/token.h"
#include <iostream>

namespace openqasm {

std::string tokenTypetoString(TokenTy ty) {
    switch (ty) {
    case TokenTy::Eof: return "EOF";
    case TokenTy::Openqasm: return "OPENQASM";
    case TokenTy::Include: return "Include";
    case TokenTy::Qreg: return "qreg";
    case TokenTy::Creg: return "creg";
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

int getBinopPrecedence(BinaryOp op) {
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