// Exception handling

#ifndef MLTOOLS_EXCEPTIONS_H
#define MLTOOLS_EXCEPTIONS_H

#include <errno.h>
#include <stdio.h>

#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TO_STRING(__LINE__) ": "

#define ERROR(msg) {perror(AT msg); exit(errno);}
#define ERRORF(fmt, ...) {fprintf(stderr, AT fmt ": %s\n", __VA_ARGS__, strerror(errno)); exit(errno);}

#endif //ifndef MLTOOLS_EXCEPTIONS_H
