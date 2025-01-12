#include "openqasm/parser.h"
#include "saot/Parser.h"

#include "utils/CommandLine.h"



static auto ArgSomeInteger = utils::cl::registerArgument<int>("some-integer");

int main(int argc, char** argv) {
  std::cerr << "Starting the main function\n";
  utils::cl::DisplayArguments();

  utils::cl::ParseCommandLineArguments(argc, argv);

  utils::cl::DisplayArguments();

  utils::cl::unregisterAllArguments();
  return 0;
}