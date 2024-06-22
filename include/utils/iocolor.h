#ifndef UTILS_IOCOLOR_H
#define UTILS_IOCOLOR_H

#include <string>

namespace Color {
    const static std::string RESET = "\033[0m";

    const static std::string BOLD = "\033[1m";

    const static std::string RED_FG = "\033[31m";
    const static std::string GREEN_FG = "\033[32m";
    const static std::string YELLOW_FG = "\033[33m";
    const static std::string BLUE_FG = "\033[34m";
    const static std::string MAGENTA_FG = "\033[35m";
    const static std::string CYAN_FG = "\033[36m";
}

#endif // UTILS_IOCOLOR_H