#include "sycl_popsift/common/debug_macros.hpp"
// #include "sycl/queue.hpp"

// #include "sycl/queue.hpp"

// #include <sycl/sycl.hpp>

// #include <sstream>

// namespace popsift {
// namespace sycl_helpers {
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
//         ss << "Memory allocation failed" << e.what();
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//     return ptr;
// }
//
// }
// }

// namespace popsift {
// namespace sycl_helpers {
//
// template<class T>
// T* malloc_devT(int num, const char* file, int line, char* error_message, sycl::queue Q)
// {
//     T* ptr;
//     try
//     {
//         ptr = sycl::malloc_device<T>(num, Q);
//     }
//     catch(const sycl::exception& e)
//     {
//         std::stringstream ss;
//         ss << error_message << e.what();
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//
// #ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
//     // popsift::cuda::memset_sync(*ptr, 0, sz, file, line);
//     // Should probably change this to function so we can have try catch
//     Q.memset(ptr, 0, num * sizeof(T));
// #endif // NDEBUG
//
//     return ptr;
// }
// }
//
// }
