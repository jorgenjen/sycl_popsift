#pragma once

// #include "sycl/queue.hpp"
// #include <sycl/queue.hpp>
// #include <sycl/sycl.hpp>

// Results in file not found and without it malloc_devT does not work hence moved to malloc_devt.hpp file
// #include <sycl/sycl.hpp>

#include <sycl/sycl.hpp>

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

namespace popsift {
namespace sycl_common {

void print_region(
  float* ptr, const char* identifier, int start_x, int end_x, int start_y, int end_y, int width, sycl::queue Q);

template<class T>
T* malloc_devT(int num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::malloc_device<T>(num, Q);
        std::stringstream ss;
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;

    // TODO: Consider adding debug option like this
    // #ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    //     popsift::cuda::memset_sync(*ptr, 0, sz, file, line);
    // #endif // NDEBUG
}

template<class T>
T* malloc_sharedT(int num, const char* file, int line, const char* error_message, sycl::queue Q)
{
    T* ptr;
    try
    {
        ptr = sycl::malloc_shared<T>(num, Q);
        std::stringstream ss;
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;

    // TODO: Consider adding debug option like this
    // #ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    //     popsift::cuda::memset_sync(*ptr, 0, sz, file, line);
    // #endif // NDEBUG
}

} // namespace sycl_common
} // namespace popsift
