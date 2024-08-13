#ifndef UTILS_IOCOLOR_H
#define UTILS_IOCOLOR_H

namespace Color {
    const static char* RESET = "\033[0m";

    const static char* BOLD = "\033[1m";
    const static char* ITATIC = "\033[3m";

    const static char* RED_FG = "\033[31m";
    const static char* GREEN_FG = "\033[32m";
    const static char* YELLOW_FG = "\033[33m";
    const static char* BLUE_FG = "\033[34m";
    const static char* MAGENTA_FG = "\033[35m";
    const static char* CYAN_FG = "\033[36m";
}

#endif // UTILS_IOCOLOR_H