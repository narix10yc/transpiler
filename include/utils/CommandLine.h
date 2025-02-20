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
    AF_Normal,     // Either '-' or '--' allowed. Value must follow '=' or space
    AF_Positional,
    AF_Prefix,     // value can follow directly
    AF_DoubleDash, // must use '--'
  };

  enum ValueFormat {
    VF_Required,
    VF_Optional,
  };

  /// @brief The base class for all parsers
  template<typename ValueType>
  class ArgumentParser;

  namespace internal {
    class ArgumentBase {
    protected:
      StringRef _name;
      StringRef _desc = "";
      ArgumentFormat _argFormat = AF_Normal;
      ValueFormat _valueFormat = VF_Required;
    public:
      ArgumentBase(StringRef name) : _name(name) {}
      virtual ~ArgumentBase() = default;

      ArgumentBase(const ArgumentBase&) = delete;
      ArgumentBase(ArgumentBase&&) = delete;
      ArgumentBase& operator=(const ArgumentBase&) = delete;
      ArgumentBase& operator=(ArgumentBase&&) = delete;

      [[nodiscard]]
      virtual bool parseValue(StringRef clValue) = 0;

      virtual void printValue(std::ostream& os) const = 0;

      std::ostream& error(std::ostream& os = std::cerr) const {
        return os << BOLDRED("Error: ")
                  << "command line argument '" << _name << "': ";
      }

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

    template<typename ClassType, typename ValueType>
    class ArgumentCRTP : public ArgumentBase {
    protected:
      using ParserType = std::function<bool(StringRef, ValueType&)>;
      ValueType _value;
      ParserType _parseFunc;

      [[nodiscard]]
      bool parseValue(StringRef clValue) override {
        return _parseFunc(clValue, _value);
      }
    public:
      ArgumentCRTP(StringRef name)
      : ArgumentBase(name), _value(), _parseFunc(ArgumentParser<ValueType>()) {
        _name = name;
        while (!name.empty() && *name.begin() == '-')
          _name.increment();
      }

      operator ValueType() const {
        return _value;
      }

      friend std::ostream& operator<<(
          std::ostream& os, const ArgumentCRTP<ClassType, ValueType>& arg) {
        return os << static_cast<const ValueType&>(arg);
      }

      void printValue(std::ostream& os) const override {
        os << _value;
      }

      ClassType& setParser(const ParserType& parser) {
        _parseFunc = parser;
        return static_cast<ClassType&>(*this);
      }

      ClassType& init(ValueType v) {
        _value = v;
        return static_cast<ClassType&>(*this);
      }

      ClassType& desc(const char* s) {
        _desc = s;
        return static_cast<ClassType&>(*this);
      }

      ClassType& setArgumentFrmat(ArgumentFormat af) {
        _argFormat = af;
        assert((af == AF_Prefix ^ _name.length() != 1) &&
               "Prefix argument must have length 1");
        return static_cast<ClassType&>(*this);
      }

      ClassType& setValueFormat(ValueFormat vf) {
        _valueFormat = vf;
        return static_cast<ClassType&>(*this);
      }

      ClassType& setArgumentPositional() {
        return setArgumentFrmat(AF_Positional);
      }

      ClassType& setArgumentPrefix() {
        return setArgumentFrmat(AF_Prefix);
      }

      ClassType& setValueRequired() {
        return setValueFormat(VF_Required);
      }
    };

  } // namespace internal


    
  template<typename ValueType>
  class ArgumentParser {
  public:
    ArgumentParser() = default;

    /// @param clValue commandline value
    /// @return false on error
    bool operator()(StringRef clValue, ValueType& valueToWriteOn);
  };

  template<typename ValueType>
  class Argument : public internal::ArgumentCRTP<Argument<ValueType>, ValueType> {
    using CRTPBase = internal::ArgumentCRTP<Argument<ValueType>, ValueType>;
  public:
    Argument(StringRef name) : CRTPBase(name) {
      // internal::ArgumentRegistry::arguments().push_back(this);
    }
  };

  using ArgString = Argument<std::string>;
  using ArgBool = Argument<bool>;
  using ArgInt = Argument<int>;
  using ArgDouble = Argument<double>;

  // template<>
  // void ArgBool::printValue(std::ostream& os) const override;

  // template <>
  // void ArgBool::printValue(std::ostream& os) const {
  //   os << (_value ? "True" : "False");
  // }

  template<typename ValueType>
  Argument<ValueType>& registerArgument(StringRef name) {
    auto* arg = new Argument<ValueType>(name);
    internal::ArgumentRegistry::arguments().push_back(arg);
    return *arg;
  }

  void unregisterAllArguments();

} // namespace utils::cl
#else
  static_assert(false, "CommandLine.h should only be included once");
#endif // COMMANDLINE_COMMANDLINE_H
