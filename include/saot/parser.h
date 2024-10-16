#ifndef SAOT_PARSER_H
#define SAOT_PARSER_H

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "saot/ast.h"
#include "saot/Polynomial.h"
#include "utils/iocolor.h"

namespace saot::ast {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int { 
    Eof = -1,
    EndOfLine = -4,
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
    case TokenTy::Numeric: return "Numeric";
    case TokenTy::Identifier: return "Identifier";
    case TokenTy::Circuit: return "'circuit'";

    case TokenTy::Add: return "Add";
    case TokenTy::Sub: return "Sub";
    case TokenTy::Mul: return "Mul";
    case TokenTy::Div: return "Div";
    case TokenTy::Equal: return "Equal";
    case TokenTy::EqualEqual: return "EqualEqual";
    case TokenTy::Less: return "Less";
    case TokenTy::LessEqual: return "LessEqual";
    case TokenTy::Greater: return "Greater";
    case TokenTy::GreaterEqual: return "GreaterEqual";

    case TokenTy::Comma: return "Comma";
    case TokenTy::Semicolon: return "Semicolon";
    case TokenTy::L_RoundBraket: return "L_RoundBraket";
    case TokenTy::R_RoundBraket: return "R_Roundbraket";
    case TokenTy::L_SquareBraket: return "L_SquareBraket";
    case TokenTy::R_SquareBraket: return "R_SquareBraket";
    case TokenTy::L_CurlyBraket: return "L_CurlyBraket";
    case TokenTy::R_CurlyBraket: return "R_CurlyBraket";
    case TokenTy::SingleQuote: return "SingleQuote";
    case TokenTy::DoubleQuote: return "DoubleQuote";
    case TokenTy::Hash: return "Hash";
    case TokenTy::Percent: return "Percent";
    case TokenTy::AtSymbol: return "AtSymbol";
    case TokenTy::Comment: return "'Comment'";
    case TokenTy::CommentStart: return "'CommentStart'";
    case TokenTy::CommentEnd: return "'CommentEnd'";

    case TokenTy::LineFeed: return "LineFeed";
    case TokenTy::CarriageReturn: return "CarriageReturn";
    case TokenTy::EndOfLine: return "EndOfLine";

    case TokenTy::Unknown: return "<Unknown>";
    default: return "'Not Implemented'";
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
    int colStart;
    int colEnd;

    Token(TokenTy type, const std::string& str, int colStart, int colEnd)
        : type(type), str(str), colStart(colStart), colEnd(colEnd) {}

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

    std::string to_string() const {
        switch (type) {
        case TokenTy::Numeric:
            return "Numeric(" + str + ")";
        case TokenTy::Identifier:
            return "Identifier(" + str + ")";
        default:
            return TokenTyToString(type);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Token& token) {
        return os << token.to_string();
    }
};

class Parser {
protected:
    int lineNumber;
    int lineLength;
    std::string currentLine;
    std::ifstream file;

    std::vector<Token> tokenVec;
    std::vector<Token>::const_iterator tokenIt;

    void throwParserError(const char* msg, std::ostream& os = std::cerr) const;
    void throwParserError(const std::string& msg, std::ostream& os = std::cerr) const {
        return throwParserError(msg.c_str(), os);
    }

    std::ostream& displayParserWarning(const char* msg, std::ostream& os = std::cerr) const;
    std::ostream& displayParserWarning(const std::string& msg, std::ostream& os = std::cerr) const {
        return displayParserWarning(msg.c_str(), os);
    }

    std::ostream& displayParserLog(const char* msg, std::ostream& os = std::cerr) const;
    std::ostream& displayParserLog(const std::string& msg, std::ostream& os = std::cerr) const {
        return displayParserLog(msg.c_str(), os);
    }


    /// @brief read one line from file. Return the length of the read line.
    /// @return -1 if EoF is reached. In such case file.close() will be called.
    /// Otherwise, return the size of tokenVec
    int readLine();
    Token parseToken(int col);

    double convertCurTokenToFloat() const {
        if (tokenIt->type != TokenTy::Numeric)
            throwParserError("Expect a float, but got " + TokenTyToString(tokenIt->type));

        int count = 0;
        for (const auto& c : tokenIt->str) {
            if (c == '.')
                count++;
        }
        if (count > 1)
            throwParserError("Unable to parse '" + tokenIt->str + "' to float");
        return std::stod(tokenIt->str);
    }

    int convertCurTokenToInt() const {
        if (tokenIt->type != TokenTy::Numeric)
            throwParserError("Expect an integer, but got " + TokenTyToString(tokenIt->type));
    
        for (const auto& c : tokenIt->str) {
            if (c == '.') {
                throwParserError("Unable to parse '" + tokenIt->str + "' to int");
                return -1;
            }
        }
        return std::stoi(tokenIt->str);
    }

    void proceed() {
        if (++tokenIt == tokenVec.cend())
            readLine();
    }

    /// @brief Proceed to the next token, ensuring a specific token type
    /// @param ty If successful, tokenIt points to a token that has type 
    void proceedWithType(TokenTy ty, bool allowLineFeed = false) {
        tokenIt++;
        if (tokenIt == tokenVec.cend()) {
            if (!allowLineFeed) {
                std::stringstream ss;
                ss << "Expecting token type " << TokenTyToString(ty) << ", "
                << "but reached end of line";
                throwParserError(ss.str());
            } else {
                readLine();
            }
        }
        if (tokenIt->type != ty) {
            std::stringstream ss;
            ss << "Expecting token type " << TokenTyToString(ty) << ", "
               << "but nextToken is " << (*tokenIt);
            throwParserError(ss.str());
        }
    }

    bool optionalProceedWithType(TokenTy ty) {
        if ((tokenIt+1) != tokenVec.cend() && (tokenIt+1)->type == ty) {
            tokenIt++;
            return true;
        }
        return false;
    }

    double _parseRealNumber();
    
    std::complex<double> _parseComplexNumber();

    QuantumCircuit& _parseCircuitBody(QuantumCircuit& qc);
    
    GateMatrix::params_t _parseParams_t();
    GateApplyStmt _parseGateApply();

    ParameterDefStmt _parseParameterDefStmt();

    saot::Polynomial _parseSaotPolynomial();
    // bool _parseStatement(RootNode&);
public:
    Parser(const std::string& fileName)
        : lineNumber(0), currentLine(""), file(fileName),
          tokenVec(), tokenIt(tokenVec.cbegin()) {}

    QuantumCircuit parse();
};




} // namespace saot::parse

#endif // SAOT_PARSER_H