#include "openqasm/parser.h"
#include "qch/ast.h"

std::string augmentFileName(std::string& fileName, std::string by) {
    auto lastDotPos = fileName.find_last_of(".");
    std::string newName;
    if (lastDotPos == std::string::npos) 
        newName = fileName;
    else 
        newName = fileName.substr(0, lastDotPos);

    return newName + by;
}


int main(int argc, char *argv[]) {

    std::string inputFilename = argv[1];

    std::cerr << "-- Input file: " << inputFilename << std::endl;

    openqasm::Parser parser(inputFilename, 0);

    std::string qchFileName = augmentFileName(inputFilename, ".qch");

    // parse and write ast
    auto qasmRoot = parser.parse();
    std::cerr << "-- qasm AST built\n";

    auto qchRoot = qasmRoot->toQch();
    std::cerr << "-- converted to qch AST\n";

    std::ofstream f(qchFileName);
    std::cerr << "-- opened " << qchFileName << "\n";

    qchRoot->print(f);

    std::cerr << "qch file written at " << qchFileName << std::endl;
    f.close();

    return 0;
}