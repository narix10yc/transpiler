#include "utils/CommandLine.h"

using namespace utils;
using namespace utils::cl::internal;

ArgumentRegistry ArgumentRegistry::Instance;

static int levenshteinDistance(StringRef s1, StringRef s2) {
  size_t len1 = s1.length(), len2 = s2.length();
  std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

  for (size_t i = 0; i <= len1; ++i) dp[i][0] = i;
  for (size_t j = 0; j <= len2; ++j) dp[0][j] = j;

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      if (s1[i - 1] == s2[j - 1])
        dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
    }
  }
  return dp[len1][len2];
}

[[noreturn]] static void terminateNoProvidingRequiredValue(StringRef name) {
  std::cerr << BOLDRED("[ERROR] ") << "Argument '" << name << "' "
            "requires a value, but is not provided.\n";
  std::exit(1);
}

[[noreturn]] static void terminateNoPositionalArgument(StringRef value) {
  std::cerr << BOLDRED("[ERROR] ") << "Argument value '" << value << "' "
              "does not match any positional argument.\n";
  std::exit(1);
}

[[noreturn]] static void terminateEmptyArgument() {
  std::cerr << BOLDRED("[ERROR] ") << "Empty argument\n";
  std::exit(1);
}

[[noreturn]] static void terminateUnknownArgument(StringRef arg) {
  std::cerr << BOLDRED("[ERROR] ") << "Unknown argument '" << arg << "'. ";
  int minDist = 100;
  StringRef bestMatchArg;
  for (const auto* a: ArgumentRegistry::arguments()) {
    StringRef candidateArg(a->getName());
    auto candidateDist = levenshteinDistance(candidateArg, arg);
    if (candidateDist < minDist) {
      minDist = candidateDist;
      bestMatchArg = candidateArg;
    }
  }

  if (minDist < 50)
    std::cerr << "Did you mean '" << bestMatchArg << "'?";
  std::cerr << "\n";
  std::exit(1);
}

static ArgumentBase* matchArgument(StringRef name) {
  // match an argument
  for (auto* a: ArgumentRegistry::arguments()) {
    StringRef candidateArg(a->getName());
    if (candidateArg.compare(name) == 0) {
      return a;
    }
  }
  return nullptr;
}

void utils::cl::ParseCommandLineArguments(int argc, char** argv) {
  std::vector<ArgumentBase*> prefixArgs;
  std::vector<ArgumentBase*> nonPrefixArgs;
  ArgumentBase* positionalArg = nullptr;
  for (auto* a : ArgumentRegistry::arguments()) {
    switch (a->getArgFormat()) {
      case AF_Positional: {
        assert(positionalArg == nullptr);
        positionalArg = a;
        break;
      }
      case AF_Prefix: {
        prefixArgs.push_back(a);
        break;
      }
      default:
        nonPrefixArgs.push_back(a);
    }
  }

  int i = 1;
  while (i < argc) {
    int nHyphens = 0;
    StringRef name(argv[i++]);
    while (*name.begin() == '-') {
      name.increment();
      nHyphens++;
    }
    if (name.empty())
      terminateEmptyArgument();

    // positional arguments
    if (nHyphens == 0) {
      if (positionalArg == nullptr)
        terminateNoPositionalArgument(name);
      positionalArg->parseValue(name);
      continue;
    }

    // split argument name by '='
    StringRef value(name);
    while (!value.empty() && *value.begin() != '=')
      value.increment();
    name = StringRef(name.begin(), value.begin() - name.begin());
    if (!value.empty()) {
      assert(*value.begin() == '=');
      value.increment();
    }

    auto* pendingArgument = matchArgument(name);
    if (pendingArgument == nullptr)
      terminateUnknownArgument(name);
    if (value.empty() && pendingArgument->getValueFormat() == VF_Required) {
      if (i == argc)
        terminateNoProvidingRequiredValue(name);
      value = StringRef(argv[i++]);
    }
    pendingArgument->parseValue(value);
  }
}

void utils::cl::DisplayArguments() {
  std::cerr << "--------------- " << ArgumentRegistry::arguments().size()
            << " Arguments ---------------\n";
  for (const auto* a: ArgumentRegistry::arguments()) {
    auto argName = a->getName();
    std::cerr << argName << ": ";
    if (argName.length() < 30)
      std::cerr << std::string(30 - argName.length(), ' ');
    a->printValue(std::cerr);
    std::cerr << "\n";
  }
}


void utils::cl::DisplayHelp() {
  assert(false && "Not Implemented");
}

void cl::unregisterAllArguments() {
  for (const auto* a: ArgumentRegistry::arguments())
    delete a;
  ArgumentRegistry::arguments().clear();
}
