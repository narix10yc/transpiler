#include "saot/NewParser.h"
#include "saot/CircuitGraph.h"


using namespace saot::parse;

void Lexer::lex(Token& tok) {
    assert(bufferBegin <= curPtr && curPtr <= bufferEnd);
    
    char c = *(curPtr++);
    while (c == ' ' || c == '\r')
        c = *(curPtr++);

    switch (c) {
    case '\0': tok = Token(tk_Eof); return;
    case '(': tok = Token(tk_L_RoundBraket); return;
    case ')': tok = Token(tk_R_RoundBraket); return;
    case '[': tok = Token(tk_L_SquareBraket); return;
    case ']': tok = Token(tk_R_SquareBraket); return;
    case '<': tok = Token(tk_Less); return;
    case '>': tok = Token(tk_Greater); return;

    case '*': {
        if (*(curPtr++) == '*')
            tok = Token(tk_Pow);
        else {
            --curPtr;
            tok = Token(tk_Mul);
        }
        return;
    }
    
    default:
        if ('0' <= c && c <= '9') {
            auto* bufferStart = curPtr - 1;
            // while(*(++))
            return;
        }
        assert(std::isalpha(c) && "Can only parse identifiers now");

        
    }
}

saot::CircuitGraph Parser::parse() {

}