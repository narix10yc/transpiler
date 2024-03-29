#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <filesystem>

#include "parser.h"
#include "ast.h"
#include "utils.h"

#ifndef EXAMPLE_DIR 
#define EXAMPLE_DIR "."
#endif


std::string augmentFileName(std::string& fileName, std::string by) {
    auto lastDotPos = fileName.find_last_of(".");
    std::string newName;
    if (lastDotPos == std::string::npos) { newName = fileName; }
    else { newName = fileName.substr(0, lastDotPos); }

    return newName + by;
}

using namespace openqasm;

int main(int argc, char *argv[]) {

    std::string inputFilename = argv[1];

    std::cerr << "-- Input file: " << inputFilename << std::endl;

    Parser parser(inputFilename, 3);

    std::string astFileName = augmentFileName(inputFilename, "_ast.txt");

    // parse and write ast
    parser.parse();
    std::ofstream f(astFileName);
    parser.prettyPrintRoot(f);
    std::cerr << "AST written at " << astFileName << std::endl;
    f.close();

    return 0;
}