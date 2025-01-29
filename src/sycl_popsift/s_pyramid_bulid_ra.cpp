#include "sycl/ext/oneapi/experimental/builtins.hpp"
#include "sycl/ext/oneapi/experimental/graph.hpp"
#include "sycl/usm.hpp"
#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <cmath>
#include <cstddef>

namespace popsift {

namespace normalizedSource {

// NOTE: Should probably be writen as a functor instead of a lambda inside of a function
// swap out the texture memory with normal memory or work-group memory (scratch pad memory/shared memory)
// and then the rest of the kernel should be the same just use nd_range and it should be the same as this
// quite straight forward (I think and HOPE!)

// __global__ static void horiz(
// static void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_w, int dst_h, float
// shift)
// {
//     // Create octave-0 - level-0 from the input image.
//     const int write_x = blockIdx.x * blockDim.x + threadIdx.x;
//     const int write_y = blockIdx.y;
//
//     if(write_x >= dst_w)
//         return;
//
//     const int span = d_gauss.dd.span[0];
//     const float* filter = &d_gauss.dd.filter[0];
//     const float read_x = (blockIdx.x * blockDim.x + threadIdx.x + shift) / dst_w;
//     const float read_y = (blockIdx.y + shift) / dst_h;
//
//     if(write_y == 10 && write_x == 10)
//     {
//         printf("span: %d\n", span);
//         printf("filter: %f\n", filter[0]);
//         printf("read_x: %f\n", read_x);
//         printf("read_y: %f\n", read_y);
//
//         // for(int i = 0; i >
//     }
//     // std::cout << "span for horiz: " << span << std::endl;
//     // std::cout << "filter for horiz: " << filter << std::endl;
//
//     float out = 0.0f;
//
// #pragma unroll
//     for(int offset = span; offset > 0; offset--)
//     {
//         const float& g = filter[offset];
//         const float offrel = float(offset) / dst_w;
//         const float v1 = tex2D<float>(src_linear_tex, read_x - offrel, read_y);
//         const float v2 = tex2D<float>(src_linear_tex, read_x + offrel, read_y);
//         out += ((v1 + v2) * g);
//         if(write_y == 10 && write_x == 10)
//         {
//             printf("\n\n");
//             printf("offset: %d\n", offset);
//             printf("g: %f\n", g);
//             printf("offrel: %f\n", offrel);
//             printf("v1: %f\n", v1);
//             printf("v2: %f\n", v2);
//             printf("out: %f\n", out);
//             printf("\n\n");
//         }
//     }
//     const float& g = filter[0];
//     const float v3 = tex2D<float>(src_linear_tex, read_x, read_y);
//     out += (v3 * g);
//
//     surf2DLayeredwrite(out * 255.0f, dst_data, write_x * 4, write_y, 0, cudaBoundaryModeZero);
// }

} // namespace normalizedSource

void Pyramid::horiz_from_input_image(const Config& conf, Image* base, sycl::event d_gauss_write)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    // dim3 block(128, 1);
    // dim3 grid;
    // grid.x = grid_divide(width, 128);
    // grid.y = height;

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    std::size_t grid_x = grid_divide(width, 128); // different from CUDA
    std::cout << "grid_x: " << grid_x << "widht: " << width << "height: " << height << std::endl;
    sycl::range global{grid_x, (size_t)height}; // not sure why heiht needs to be size_t
    sycl::range local{128, 1};
    std::cout << "Global range: "
              << "dim0 = " << global.get(0) << ", "
              << "dim1 = " << global.get(1) << "\n";

    std::cout << "Local range: "
              << "dim0 = " << local.get(0) << ", "
              << "dim1 = " << local.get(1) << "\n";

    std::cout << "H gauss" << h_gauss.required_filter_stages << std::endl;

    unsigned char* intermediate; // needs to be moved to the octave so that it is releated to it and working properly

    try
    {
        intermediate = sycl::malloc_device<unsigned char>(width * height, _device_queue);
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }

    _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(d_gauss_write);
        auto gauss_ptr = _d_gauss; // needed to avoid implicitly capturing this which is not allowed
        auto input = base->getInput();

        const int span = _d_gauss->dd.span[0];
        const float* filter = &_d_gauss->dd.filter[0];

        sycl::stream stream_out(1024, 256, cgh); // for debugging

        cgh.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
            int x = it.get_global_id(0);
            int y = it.get_global_id(1);
            sycl::range gr = it.get_global_range();
            sycl::range lr = it.get_local_range();

            // could have two different kernels one with this and one without
            // depending on if it is perfectly divisible by 128 but might not be worth it... Test
            if(x >= width)
                return;

            // The code

            int read_x = x + shift;
            int read_y = y + shift;

            float out = 0.0f;

            // the code end

            // if(x == 1279 && y == 851) // final work item for 1280 x 851 image
            if(x == 0 && y == 0)
            {
                // stream_out << "\n\n\t\tHello sycl! (" << x << ", " << y << ")" << sycl::endl;
                stream_out << "\t\tglobal range: " << gr.get(0) << " ; " << gr.get(1) << sycl::endl;
                stream_out << "\t\tlocal range: " << lr.get(0) << " ; " << lr.get(1) << sycl::endl;
                // // stream_out << "\t\tGauus stufus: " << me_gauss->required_filter_stages << sycl::endl;
                // stream_out << "\t\tGauus stufus: " << gauss_ptr->required_filter_stages << sycl::endl;
                // stream_out << "\t\tspan: " << span << sycl::endl;
                sycl::ext::oneapi::experimental::printf(
                  "\t\tfilter: %f %f %f %f %f %f\n", filter[6], filter[5], filter[4], filter[3], filter[2], filter[1]);

                stream_out << "\n\n\n";
                sycl::ext::oneapi::experimental::printf("\n\n");
                for(int y = 0; y < 10; ++y)
                {
                    for(int x = 0; x < 10; ++x)
                    {
                        // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
                        sycl::ext::oneapi::experimental::printf("%06.2f ", input[x + y * (width)] * 255.0f);
                    }
                    sycl::ext::oneapi::experimental::printf("\n");
                }
                sycl::ext::oneapi::experimental::printf("\n\n");
            }
            // if(x < 5 && y < 5)
            //     sycl::ext::oneapi::experimental::printf("\t\tPixel val(%d, %d): %f\n", x, y, input[x + y * width]);
        });
    });
    _device_queue.wait(); // temporary waiting here remove in future

    // _device_queue.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
    //     int j = it.get_global_id(0);
    //     int i = it.get_global_id(1);
    //     // const int span = this->_d_gauss->dd.span[0];
    //     // const float* filter = &this->_d_gauss->dd.filter[0];
    //
    //     if(i == 10 && j == 10)
    //     {
    //         printf("_d_gauss filter_stages %d", _d_gauss->required_filter_stages);
    //
    //         // printf("span: %d\n", span);
    //         // printf("filter: %f %f %f %f %f %f\n", filter[6], filter[5], filter[4], filter[3], filter[2],
    //         filter[1]);
    //     }
    //
    //     // for(int k = 0; k < N; ++k)
    //     // {
    //     //     c[j][i] += a[j][k] * b[k][i];
    //     // }
    // });

    // NOTE: Is an error check after kernel that is conditionally set bu an ifdef
    // consider implementing something similar
    // POP_SYNC_CHK;
}

} // namespace popsift
