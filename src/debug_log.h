#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#ifdef DEBUG
    #include <iostream>
    #define DEBUG_LOG(x) std::cout << x << std::endl
#else
    #define DEBUG_LOG(x)
#endif

#endif
