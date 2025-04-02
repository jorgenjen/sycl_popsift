#include "sycl_popsift/common/debug_macros.hpp"

#include "sycl/queue.hpp"

namespace popsift {
namespace sycl_common {

void print_region(
  float* ptr, const char* identifier, int start_x, int end_x, int start_y, int end_y, int width, sycl::queue Q)
{
    int str_len = std::strlen(identifier) + 1;
    fprintf(stderr, "hello\n");

    char* dev_msg = malloc_devT<char>(str_len, __FILE__, __LINE__, "Could not allocate print identifier", Q);
    Q.memcpy(dev_msg, identifier, (size_t)str_len).wait();

    Q.single_task([=]() {
         sycl::ext::oneapi::experimental::printf(
           "\n\n%s -- Region: y(%d -> %d) x(%d -> %d) \n", dev_msg, start_x, end_x, start_y, end_y);
         for(int y = start_y; y < end_y; ++y)
         {
             for(int x = start_x; x < end_x; ++x)
             {
                 sycl::ext::oneapi::experimental::printf("%010.6f ", ptr[x + y * (width)]);
             }
             sycl::ext::oneapi::experimental::printf("\n");
         }
         sycl::ext::oneapi::experimental::printf("\n\n");
     })
      .wait();

    sycl::free(dev_msg, Q);
    // With stream
    // Q.submit([&](sycl::handler& h) {
    //      sycl::stream out(4096, 512, h);
    //
    //      h.single_task([=]() {
    //          out << "\n"
    //              << dev_msg << " -- Region: y(" << start_y << " -> " << end_y << ") x(" << start_x << " -> " << end_x
    //              << ")\n";
    //
    //          for(int y = start_y; y < end_y; ++y)
    //          {
    //              for(int x = start_x; x < end_x; ++x)
    //              {
    //                  out << ptr[x + y * width] << " ";
    //              }
    //              out << "\n";
    //          }
    //          out << "\n\n";
    //      });
    //  })
    //   .wait();
}
}
}

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
