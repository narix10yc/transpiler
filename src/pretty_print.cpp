#include "ast.h"
#include "utils.h"

using namespace ast;

void RootNode::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "RootNode" << "\n";
    f << std::string(2*depth+2, ' ') << "(Statements:)" << "\n";
    for (const auto& item : stmts) {
        item->prettyPrint(f, depth+1);
    }
}

void IfThenElseStmt::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "IfThenElse" << "\n";
    f << std::string(2*depth+2, ' ') << "(If:)" << "\n";
    ifExpr->prettyPrint(f, depth+1);
    f << std::string(2*depth+2, ' ') << "(Then:)" << "\n";
    for (const auto& item : thenBody) {
        item->prettyPrint(f, depth+1);
    }
    f << std::string(2*depth+2, ' ') << "(Else:)" << "\n";
    for (const auto& item : elseBody) {
        item->prettyPrint(f, depth+1);
    }
}

void VersionStmt::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "Version: " << getVersion() << "\n";
}

void NumericExpr::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "Numeric[" << value << "]";
    f << "\n";
}

void VariableExpr::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "Variable[" << name << "]";
    f << "\n";
}

void UnaryExpr::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "UnaryExpr" << "\n";
}

void BinaryExpr::prettyPrint(std::ofstream &f, int depth) const {
    f << std::string(2*depth, ' ') << "BinaryExpr" << "\n";
    f << std::string(2*depth+2, ' ') << "(Op:) " << BinaryOpToString(op) << "\n";
    f << std::string(2*depth+2, ' ') << "(LHS:)" << "\n";
    lhs->prettyPrint(f, depth+1);
    f << std::string(2*depth+2, ' ') << "(RHS:)" << "\n";
    rhs->prettyPrint(f, depth+1);
}

