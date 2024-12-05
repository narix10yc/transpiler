#ifndef UTILS_IOCOLOR_H
#define UTILS_IOCOLOR_H

#define RED(MSG) "\033[31m" << MSG << "\033[0m"
#define BOLDRED(MSG) "\033[1m\033[31m" << MSG << "\033[0m"

#define GREEN(MSG) "\033[32m" << MSG << "\033[0m"
#define BOLDGREEN(MSG) "\033[1m\033[32m" << MSG << "\033[0m"

namespace IOColor {
static const char* RESET = "\033[0m";

static const char* BOLD = "\033[1m";
static const char* ITATIC = "\033[3m";

static const char* DEFAULT_FG = "\033[30m";
static const char* RED_FG = "\033[31m";
static const char* GREEN_FG = "\033[32m";
static const char* YELLOW_FG = "\033[33m";
static const char* BLUE_FG = "\033[34m";
static const char* MAGENTA_FG = "\033[35m";
static const char* CYAN_FG = "\033[36m";
} // namespace IOColor

#endif // UTILS_IOCOLOR_H