#pragma once

// #include "sycl/queue.hpp"
// #include <sycl/queue.hpp>
// #include <sycl/sycl.hpp>

// Results in file not found and without it malloc_devT does not work hence moved to malloc_devt.hpp file
// #include <sycl/sycl.hpp>

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// namespace popsift {
// namespace sycl_helpers {
// template<class T>
// T* malloc_devT(int num, const char* file, int line, sycl::queue Q);
// }
// }
//
#define POP_FATAL(s)                                                                                                   \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << __FILE__ << ":" << __LINE__ << std::endl << "    " << s;                                                 \
        throw std::runtime_error{ss.str()};                                                                            \
    }

// Not sure if i need the following ones
#define POP_FATAL_FL(s, file, line)                                                                                    \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << file << ":" << line << std::endl << "    " << s << std::endl;                                            \
        throw std::runtime_error{ss.str()};                                                                            \
    }

#define POP_CHECK_NON_NULL(ptr, s)                                                                                     \
    if(ptr == 0)                                                                                                       \
    {                                                                                                                  \
        POP_FATAL_FL(s, __FILE__, __LINE__);                                                                           \
    }

#define POP_CHECK_NON_NULL_FL(ptr, s, file, line)                                                                      \
    if(ptr == 0)                                                                                                       \
    {                                                                                                                  \
        POP_FATAL_FL(s, file, line);                                                                                   \
    }

// Cannot include sycl.hpp here and hence can't find sycl::queu so can't have the function here IDK why that is so it is
// moved to ../malloc_devt.hpp
// namespace popsift::common_sycl {
//
// template<class T>
// T* malloc_devT(int num, const char* file, int line, sycl::queue Q)
// {
//     T* ptr;
//     try
//     {
//         ptr = sycl::malloc_device<T>(num, Q);
//     }
//     catch(const sycl::exception& e)
//     {
//         std::stringstream ss;
//         ss << "Memory allocation failed: " << e.what();
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//     return ptr;
// }
//
// } // namespace popsift::common_sycl
