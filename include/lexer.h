#ifndef LEXER_H_
#define LEXER_H_

#include <iostream>
#include <fstream>
#include <queue>
#include "token.h"



class Lexer {
    std::ifstream file;
    std::queue<int> charBuf;
    bool waitFlag = false;
    int curChar;
public:
    Lexer(std::string& fileName) : file(fileName) {}
    ~Lexer() { file.close(); }

    int peekChar();

    Token getToken();

    void logError(const std::string& msg) const {
        std::cerr << "== Lexer Error == " << msg << "\n";
    }

    bool checkFileOpen() const { return file.is_open(); }

private:
    void nextChar() {
        if (charBuf.empty())
            curChar = file.get();
        else {
            curChar = charBuf.front();
            charBuf.pop();
        }
    }

    void skipToEndOfLine() {
        do { nextChar(); }
        while (curChar != EOF && curChar != '\n' && curChar != '\r');
    }
    
    Token TokenizeNumeric();
    Token TokenizeIdentifier();
};


#endif // LEXER_H_