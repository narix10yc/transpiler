#ifndef SAOT_NEWPARSER_H
#define SAOT_NEWPARSER_H

#include <fstream>
#include <cassert>

namespace saot {
    class CircuitGraph;
}

namespace saot::parse {

enum TokenKind : int {
    tk_Eof = -1,
    tk_Identifier = -2,
    tk_Numeric = -3,

    // keywords
    tk_Circuit = -10,

    // operators
    tk_Add = -30,                  // +
    tk_Sub = -31,                  // -
    tk_Mul = -32,                  // *
    tk_Div = -33,                  // /
    tk_Pow = -34,                  // **
    tk_Greater = -35,              // >
    tk_Less = -36,                 // <
    tk_Equal = -37,                // =
    tk_GreaterEqual = -38,         // >=
    tk_LessEqual = -39,            // <=
    tk_EqualEqual = -40,           // ==

    // symbols
    tk_Comma = -104,               // ,
    tk_Semicolon = -105,           // ;
    tk_L_RoundBraket = -106,       // (
    tk_R_RoundBraket = -107,       // )
    tk_L_SquareBraket = -108,      // [
    tk_R_SquareBraket = -109,      // ]
    tk_L_CurlyBraket = -112,       // {
    tk_R_CurlyBraket = -113,       // }
    tk_SingleQuote = -114,         // '
    tk_DoubleQuote = -115,         // "
    tk_AtSymbol = -116,            // @
    tk_Percent = -117,             // %
    tk_Hash = -118,                // #
    tk_Backslash = -119,           // '\'
    tk_Comment = -120,             // '//'
    tk_CommentStart = -121,        // '/*'
    tk_CommentEnd = -122,          // '*/'

    tk_LineFeed = 10, // '\n'
    tk_CarriageReturn = 13, // '\r'

    tk_Unknown = -1000,
    tk_Any = -1001,
};

class Token {
public:
    TokenKind kind;
    const char* memRefBegin;
    const char* memRefEnd;

    Token(TokenKind kind)
        : kind(kind), memRefBegin(nullptr), memRefEnd(nullptr) {}
        
    Token(TokenKind kind, const char* memRefBegin, const char* memRefEnd)
        : kind(kind), memRefBegin(memRefBegin), memRefEnd(memRefEnd) {}
    

};

class Lexer {
public:
    const char* bufferBegin;
    const char* bufferEnd;
    size_t bufferLength;

    const char* curPtr;

    Lexer(const char* fileName) {
        std::ifstream file(fileName);
        assert(file.is_open());

        bufferLength = file.tellg();
        bufferBegin = new char[bufferLength];
        bufferEnd = bufferStart + bufferLength;

        curPtr = bufferStart;
    }

    void lex(Token& tok);
};




class Parser {
    Lexer lexer;

public:
    Parser(const char* fileName) : lexer(fileName) {}

    CircuitGraph parse();

};

} // namespace saot::parse

#endif // SAOT_NEWPARSER_H