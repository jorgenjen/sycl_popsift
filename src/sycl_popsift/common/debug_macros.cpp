// #include "sycl_popsift/common/debug_macros.hpp"
// // #include "sycl/queue.hpp"
//
// // #include "sycl/queue.hpp"
//
// // #include <sycl/sycl.hpp>
//
// // #include <sstream>
//
// template<class T>
// T* malloc_devT(int num, const char* file, int line, const char* error_message, sycl::queue Q)
// {
//     T* ptr;
//     try
//     {
//         ptr = sycl::malloc_device<T>(num, Q);
//         std::stringstream ss;
//     }
//     catch(const sycl::exception& e)
//     {
//         std::stringstream ss;
//         ss << error_message << e.what();
//         std::string error_msg = ss.str(); // seems to be required to have given message show up
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//     return ptr;
// }
//
// template<class T>
// T* malloc_sharedT(int num, const char* file, int line, const char* error_message, sycl::queue Q)
// {
//     T* ptr;
//     try
//     {
//         ptr = sycl::malloc_shared<T>(num, Q);
//         std::stringstream ss;
//     }
//     catch(const sycl::exception& e)
//     {
//         std::stringstream ss;
//         ss << error_message << e.what();
//         std::string error_msg = ss.str(); // seems to be required to have given message show up
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//     return ptr;
// }
