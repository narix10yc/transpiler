#ifndef LEXER_H_
#define LEXER_H_

#include <iostream>
#include <fstream>
#include <queue>
#include "token.h"

namespace openqasm {

class Lexer {
    std::string fileName;
    std::queue<int> charBuf;
    bool waitFlag = false;
    int curChar;
    std::ifstream file;
public:
    Lexer(std::string& fileName) : fileName(fileName) {}
    ~Lexer() { file.close(); }

    int peekChar();

    Token getToken();

    void logError(const std::string& msg) const {
        std::cerr << "== Lexer Error == " << msg << "\n";
    }

    bool openFile() { 
        file = std::ifstream(fileName);
        return file.is_open();
    }

    void closeFile() { file.close(); }

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
    
    Token tokenizeNumeric();
    Token tokenizeIdentifier();
};

} // namespace openqasm

#endif // LEXER_H_