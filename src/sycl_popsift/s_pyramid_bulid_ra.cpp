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

    float* intermediate; // needs to be moved to the octave so that it is releated to it and working properly

    try
    {
        intermediate = sycl::malloc_device<float>(width * height, _device_queue);
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

            // shift would make this odd but could be detrimendal for future kernels that needs it ... so might have to
            // modify the original shift to be 0 as this is dealt with in the creating of the input image
            // int read_x = x
            // + shift; int read_y = y + shift;

            float out = 0.0f;

#pragma unroll
            for(int offset = span; offset > 0; offset--)
            {
                const float& g = filter[offset];
                // const float offrel = float(offset) / width;
                // const float v1 = tex2D<float>(src_linear_tex, read_x - offrel, read_y);
                // const float v2 = tex2D<float>(src_linear_tex, read_x + offrel, read_y);
                const auto v1_pos = x - offset;
                const auto v2_pos = x + offset;

                // clamp to left for - and clamp to right for + // does the smae as cudaAddressModeClamp for textures in
                // cuda used in popsift
                const float v1 = v1_pos < 0 ? input[y * width] : input[v1_pos + y * width];
                const float v2 = v2_pos >= width ? input[width - 1 + y * width] : input[v2_pos + y * width];
                // const float v2 = v2_pos >= width ? [width - 1 + y * width] : input[v1_pos + y * width];

                // const float v1 = tex2D<float>(src_linear_tex, read_x - offrel, read_y);
                // const float v2 = tex2D<float>(src_linear_tex, read_x + offrel, read_y);
                if(x == 0 && y == 0)
                {
                    sycl::ext::oneapi::experimental::printf("offset: %d v1=%f v2=%f\n ", offset, v1, v2);
                }
                out += ((v1 + v2) * g);
            }
            const float& g = filter[0];
            const float v3 = input[x + y * width];
            out += (v3 * g);

            // surf2DLayeredwrite(out * 255.0f, dst_data, write_x * 4, write_y, 0, cudaBoundaryModeZero);

            intermediate[x + y * width] = out * 255.0f;

            // the code end

            // if(x == 1279 && y == 851) // final work item for 1280 x 851 image
            if(x == 0 && y == 0)
            {
                sycl::ext::oneapi::experimental::printf("\n\n");
                for(int y = 0; y < 12; ++y)
                {
                    for(int x = 0; x < 12; ++x)
                    {
                        // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
                        sycl::ext::oneapi::experimental::printf("%06.2f ", input[x + y * (width)] * 255.0f);
                    }
                    sycl::ext::oneapi::experimental::printf("\n");
                }
                sycl::ext::oneapi::experimental::printf("\n\n");
            }
        });
    });
    _device_queue.wait(); // temporary waiting here remove in future

    printf("Print intermediate \n");
    _device_queue.submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() {
            sycl::ext::oneapi::experimental::printf("\n\n");
            for(int y = 0; y < 12; ++y)
            {
                for(int x = 0; x < 12; ++x)
                {
                    // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
                    sycl::ext::oneapi::experimental::printf("%06.2f ", intermediate[x + y * (width)]);
                }
                sycl::ext::oneapi::experimental::printf("\n");
            }
            sycl::ext::oneapi::experimental::printf("\n\n");
        });
    });

    _device_queue.wait();
    // print out intermediate here to see that it works like it should !

    sycl::free(intermediate, _device_queue);

    // NOTE: Is an error check after kernel that is conditionally set bu an ifdef
    // consider implementing something similar
    // POP_SYNC_CHK;
}

} // namespace popsift
