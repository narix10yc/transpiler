#ifndef COMMANDLINE_COMMANDLINE_H
#define COMMANDLINE_COMMANDLINE_H

#include "utils/StringRef.h"
#include "utils/PODVector.h"
#include "utils/iocolor.h"

namespace utils::cl {

  void ParseCommandLineArguments(int argc, char** argv);

  void DisplayArguments();

  void DisplayHelp();

  enum ArgumentFormat {
    AF_Normal,     // either '-' or '--' allowed. value must follow '=' or space
    AF_Positional,
    AF_Prefix,     // value can follow directly
    AF_DoubleDash, // must use '--'
  };

  enum ValueFormat {
    VF_Required,
    VF_Optional,
  };


  namespace internal {
    class ArgumentParser;

    class ArgumentBase {
    protected:
      StringRef _name;
      StringRef _desc = "";
      ArgumentFormat _argFormat = AF_Normal;
      ValueFormat _valueFormat = VF_Required;
    public:
      ArgumentBase(StringRef name) : _name(name) {}
      virtual ~ArgumentBase() = default;

      // ArgumentBase(const ArgumentBase&) = delete;

      virtual void parseValue(StringRef) = 0;

      virtual void printValue(std::ostream&) const = 0;

      StringRef getName() const { return _name; }
      StringRef getDesc() const { return _desc; }
      ArgumentFormat getArgFormat() const { return _argFormat; }
      ValueFormat getValueFormat() const { return _valueFormat; }
    };

    class ArgumentRegistry {
      std::vector<ArgumentBase*> _arguments;
    public:
      static ArgumentRegistry Instance;
      static std::vector<ArgumentBase*>& arguments() {
        return Instance._arguments;
      }
    };

    template<typename ClassType, typename ArgType>
    class ArgumentCRTP : public ArgumentBase {
    protected:
      ArgType _value;
    public:
      ArgumentCRTP(StringRef name) : ArgumentBase(name), _value() {
        _name = name;
        while (!name.empty() && *name.begin() == '-')
          _name.increment();
      }

      operator ArgType() const {
        return _value;
      }

      friend std::ostream& operator<<(
          std::ostream& os, const ArgumentCRTP<ClassType, ArgType>& arg) {
        return os << static_cast<const ArgType&>(arg);
      }

      void printValue(std::ostream& os) const override {
        os << _value;
      }

      ClassType& init(ArgType v) {
        _value = v;
        return static_cast<ClassType&>(*this);
      }

      ClassType& desc(const char* s) {
        _desc = s;
        return static_cast<ClassType&>(*this);
      }

      ClassType& format(ArgumentFormat f) {
        _argFormat = f;
        assert((f == AF_Prefix ^ _name.length() != 1) && "Prefix");
        return static_cast<ClassType&>(*this);
      }
    };

  } // namespace internal

template<typename ArgType>
class Argument : public internal::ArgumentCRTP<Argument<ArgType>, ArgType> {
  using CRTPBase = internal::ArgumentCRTP<Argument<ArgType>, ArgType>;
public:
  Argument(StringRef name) : CRTPBase(name) {}

  void parseValue(StringRef s) override;
};

using ArgString = Argument<std::string>;
using ArgBool = Argument<bool>;
using ArgInt = Argument<int>;
using ArgDouble = Argument<double>;

template<>
inline void ArgString::parseValue(StringRef s) {
  _value = static_cast<std::string>(s);
}

template<>
inline void ArgInt::parseValue(StringRef s) {
  _value = std::stoi(static_cast<std::string>(s));
}

template<>
inline void ArgDouble::parseValue(StringRef s) {
  _value = std::stod(static_cast<std::string>(s));
}

template<>
inline void ArgBool::parseValue(StringRef s) {
  if (s.empty())
    _value = true;
  else if (s.compare_ci("false") || s.compare_ci("off"))
    _value = false;
  else if (s.compare_ci("true") || s.compare_ci("on"))
    _value = true;
  else if (s.compare("0"))
    _value = false;
  else if (s.compare("1"))
    _value = true;
  else {
    std::cerr << BOLDRED("Error: ") << "Illegal value '" << s << "' "
              " for boolean argument " << _name << "\n";
    std::exit(1);
  }
}


template<typename ArgType>
Argument<ArgType>& registerArgument(const char* name) {
  auto* arg = new Argument<ArgType>(name);
  internal::ArgumentRegistry::arguments().push_back(arg);
  return *arg;
}

void unregisterAllArguments();

} // namespace utils::cl
#else
  static_assert(false, "CommandLine.h should only be included once");
#endif // COMMANDLINE_COMMANDLINE_H
