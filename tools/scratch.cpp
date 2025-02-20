#include "utils/CommandLine.h"

using namespace utils;

cl::ArgInt ArgMyInteger("hello");

ArgMyInteger.desc("This is my integer");

int main(int argc, char** argv) {

  utils::cl::ParseCommandLineArguments(argc, argv);

  utils::cl::DisplayArguments();



  
  return 0;
}