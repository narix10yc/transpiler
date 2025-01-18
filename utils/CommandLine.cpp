#include "utils/CommandLine.h"

using namespace utils;
using namespace utils::cl::internal;

ArgumentRegistry ArgumentRegistry::Instance;

static int levenshteinDistance(StringRef s1, StringRef s2) {
  size_t len1 = s1.length(), len2 = s2.length();
  std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

  for (size_t i = 0; i <= len1; ++i)
    dp[i][0] = i;
  for (size_t j = 0; j <= len2; ++j)
    dp[0][j] = j;

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

[[noreturn]] static void terminateDisplayHelp() {
  cl::DisplayHelp();
  std::exit(0);
}

static void warnTooManyHyphens(int nHyphens, StringRef name) {
  std::cerr << BOLDYELLOW("[Warning] ") << "Argument '" << name << "' "
            "is prefixed by " << nHyphens << " (> 2) hyphens.\n";
}

static void warnEqualSignInPrefixArg(char prefix) {
  std::cerr << BOLDYELLOW("[Warning] ") << "Prefix argument '" << prefix << "' "
            "should not be followed by '='.\n";
}

[[noreturn]] static void terminateLeadingEqualSign() {
  std::cerr << BOLDRED("[ERROR] ") << "Leading equal sign. "
            "Did you add a redundant space before it?\n";
  std::exit(1);
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

[[noreturn]] static void terminateEmptyValue(StringRef name) {
  std::cerr << BOLDRED("[ERROR] ") << "Argument '" << name << "' "
            "has no value.\n";
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

  if (minDist < 6)
    std::cerr << "Did you mean '" << bestMatchArg << "'?";
  std::cerr << "\n";
  std::exit(1);
}

[[noreturn]] static void terminateSpaceAfterEqualSign(
    StringRef name, StringRef value) {
  std::cerr << BOLDRED("[ERROR] ") << "Argument-value pair '" << name << "= "
            << value << "' should not have space after the '=' sign.\n";
  std::exit(1);
}

namespace {
  struct ArgContainer {
    ArgumentBase* arg;
    StringRef name;
    explicit ArgContainer(ArgumentBase* a) : arg(a), name(a->getName()) {}
  };
}

void utils::cl::ParseCommandLineArguments(int argc, char** argv) {
  std::vector<ArgContainer> prefixArgs;
  std::vector<ArgContainer> nonPrefixArgs;

  ArgumentBase* positionalArg = nullptr;
  for (auto* a : ArgumentRegistry::arguments()) {
    switch (a->getArgFormat()) {
      case AF_Positional: {
        assert(positionalArg == nullptr);
        positionalArg = a;
        break;
      }
      case AF_Prefix: {
        prefixArgs.emplace_back(a);
        assert(a->getName().length() == 1 &&
          "Prefix arguments must have name with length 1");
        break;
      }
      default:
        nonPrefixArgs.emplace_back(a);
    }
  }

  const auto matchPrefixArg = [&](char prefix) -> ArgumentBase* {
    for (const auto& c : prefixArgs) {
      if (*c.name.begin() == prefix)
        return c.arg;
    }
    return nullptr;
  };

  const auto matchNonPrefixArg = [&](StringRef name) -> ArgumentBase* {
    for (const auto& c : nonPrefixArgs) {
      if (c.name.compare(name) == 0)
        return c.arg;
    }
    return nullptr;
  };

  int i = 1;

  // const auto grabNext = [&]() -> StringRef {
    // if (i == argc) {
      // terminate
    // }
  // };

  while (i < argc) {
    StringRef name(argv[i++]);
    assert(!name.empty());
    if (*name.begin() == '=') {
      terminateLeadingEqualSign();
      // terminated
    }
    int nHyphens = 0;
    while (*name.begin() == '-') {
      name.increment();
      nHyphens++;
    }
    if (name.empty()) {
      terminateEmptyArgument();
      // terminated
    }

    if (nHyphens > 2)
      warnTooManyHyphens(nHyphens, name);

    if (name.compare("help") == 0) {
      terminateDisplayHelp();
      // terminated
    }

    // positional arguments
    if (nHyphens == 0) {
      if (positionalArg == nullptr) {
        terminateNoPositionalArgument(name);
        // terminated
      }
      positionalArg->parseValue(name);
      continue;
    }

    // split argument name by '='. value is empty <=> there is no '=' sign
    StringRef value(name);
    while (!value.empty() && *value.begin() != '=')
      value.increment();
    name = StringRef(name.begin(), value.begin() - name.begin());
    if (!value.empty()) {
      assert(*value.begin() == '=');
      value.increment();
      if (value.empty()) {
        terminateSpaceAfterEqualSign(name, (i < argc) ? argv[i+1] : "");
        // terminated
      }
    }

    ArgumentBase* pendingArg = nullptr;
    // prefix arguments
    if ((pendingArg = matchPrefixArg(*name.begin()))) {
      if (!value.empty()) {
        warnEqualSignInPrefixArg(*name.begin());
        pendingArg->parseValue(value);
        continue;
      }
      if (name.length() == 1) {
        if (i == argc) {
          terminateNoProvidingRequiredValue(name);
          // terminated
        }
        pendingArg->parseValue(argv[i++]);
        continue;
      }
      // name.length() > 1
      name.increment();
      pendingArg->parseValue(name);
      continue;
    }

    pendingArg = matchNonPrefixArg(name);
    if (pendingArg == nullptr)
      terminateUnknownArgument(name);
    if (value.empty() && pendingArg->getValueFormat() == VF_Required) {
      if (i == argc)
        terminateNoProvidingRequiredValue(name);
      value = StringRef(argv[i++]);
    }
    pendingArg->parseValue(value);
  }
}

void utils::cl::DisplayArguments() {
  std::cerr << "----------------- " << ArgumentRegistry::arguments().size()
            << " Arguments -----------------\n";
  for (const auto* a: ArgumentRegistry::arguments()) {
    auto argName = a->getName();
    std::cerr << argName << ": ";
    if (argName.length() < 30)
      std::cerr << std::string(30 - argName.length(), ' ');
    a->printValue(std::cerr);
    std::cerr << "\n";
  }
  std::cerr << "--------------- End of Arguments ---------------\n";
}


void utils::cl::DisplayHelp() {
  std::cerr << "<-help>\n";
}

void cl::unregisterAllArguments() {
  for (const auto* a: ArgumentRegistry::arguments())
    delete a;
  ArgumentRegistry::arguments().clear();
}
