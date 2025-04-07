#include "sycl_popsift/common/debug_macros.hpp"

// #include "sycl/queue.hpp"
// #include "sycl/sycl.hpp"

namespace popsift {
namespace sycl_common {

void print_region(
  float* ptr, const char* identifier, int start_x, int end_x, int start_y, int end_y, int width, sycl::queue Q)
{
    int str_len = std::strlen(identifier) + 1;

    // char* dev_msg = malloc_devT<char>(str_len, __FILE__, __LINE__, "Could not allocate print identifier", Q);
    char* dev_msg = sycl::malloc_device<char>(str_len, Q);
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
