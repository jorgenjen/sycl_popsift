#pragma once

// #include "sycl/queue.hpp"
// #include <sycl/queue.hpp>
// #include <sycl/sycl.hpp>

// Results in file not found and without it malloc_devT does not work hence moved to malloc_devt.hpp file
// #include <sycl/sycl.hpp>
// #include "sycl/queue.hpp"

#include <sycl/sycl.hpp>

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

// TODO: Move to cpp file keep declarations
// NOTE: Might be loosing some performance by passing custom error message to each malloc... PopSift does not do that so
// mby  I should not have done that for fair comparison (but should not matter much I think :D)

void print_region(
  float* ptr, const char* identifier, int start_x, int end_x, int start_y, int end_y, int width, sycl::queue Q);

// BUG: Modify all malloc functions to check if it is nullptr not catch exeptions. all sycl allocators does not throw
// exception when they cant allocate they only return nullptr (Should only not work when out that type of memory)
template<typename T>
T* malloc_devT(int num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::malloc_device<T>(num, Q);
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

template<typename T>
T* alloc_aligned_deviceT(
  size_t alignment, size_t num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::aligned_alloc_device<T>(alignment, num, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        // TODO: Verify that this line is needed makes no sense...
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;
}

template<typename T>
T* malloc_sharedT(size_t num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::malloc_shared<T>(num, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;
}

// TODO: Add pinned memory version for host meomory (is an extension)
template<typename T>
T* malloc_hostT(size_t num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::malloc_host<T>(num, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;
}

template<typename T>
T* alloc_aligned_hostT(
  size_t alignment, size_t num, const char* file, int line, const char* error_message, sycl::queue& Q)
{
    T* ptr;
    try
    {
        ptr = sycl::aligned_alloc_host<T>(alignment, num, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
    return ptr;
}

} // namespace sycl_common

namespace common {

template<typename T>
T* new_hostT(int num, const char* file, int line, const char* error_message)
{
    try
    {
        // Only returns if new is success full
        return new T[num];
    }
    catch(const std::bad_alloc& e)
    {
        std::stringstream ss;
        ss << error_message << e.what();
        std::string error_msg = ss.str(); // seems to be required to have given message show up
        POP_FATAL_FL(ss.str(), file, line);
    }
}

} // namespace common

} // namespace popsift
