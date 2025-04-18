#include "s_image.hpp"

#include "common/bindless_helpers.hpp"
#include "common/debug_macros.hpp"

#include <sycl/sycl.hpp>

#include <cmath> // ceilf
#include <cstdio>
#include <iostream>

namespace popsift {

Image::Image(sycl::queue Q)
  : _w(0)
  , _h(0)
  , _max_w(0)
  , _max_h(0)
  , _device_queue(Q)
{}

Image::Image(int w, int h, sycl::queue Q)
  : _w(w)
  , _h(h)
  , _max_w(w)
  , _max_h(h)
  , _device_queue(Q)
{
    // Not sure if using w and h is correct need to refactor the whole scaling thing as it is kinda confusing
    // Should probably just use the scaled size and nothing else when using USM
    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      w * h, __FILE__, __LINE__, "Could not allocate memory for image on device", Q);

    _device_img = popsift::sycl_common::malloc_devT<float>(
      w * h, __FILE__, __LINE__, "Could not allocate memory for float representation of image on device", Q);
}

Image::~Image()
{
    fprintf(stderr, "\n\tDESTROYING IMAGE\n");
    if(_max_w == 0)
        return;

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);

    if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
    {
        fprintf(stderr, "DESTROYING HANDLE AND UNDERLYING OBJECT FOR BINDLESS IMAGE\n");
        // Free bindless image
        namespace syclexp = sycl::ext::oneapi::experimental;
        syclexp::free_image_mem(_dev_img_handle, sycl::ext::oneapi::experimental::image_type::standard, _device_queue);
        syclexp::destroy_image_handle(_sampled_dev_img_handle, _device_queue);
    }

    // destroyTexture( );
    // _input_image_d.freeDev( );
    // _input_image_h.freeHost( popsift::CudaAllocated );
}

// Modified using sclaed and not scaled a bit confusing and ugly so should  refator if this is part of final
// void Image::resetDimensions(int w, int h, int scaled_w, int scaled_h)
void Image::resetDimensions(int w, int h, float upscaleFactor)
{
    if(_max_w == 0 && _max_h == 0)
    {
        // First time instantiating
        _max_w = _w = w;
        _max_h = _h = h;

        if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
        {
            // upscaleFactor is unused in this case
            allocate_bindless();
            allocate_usm(upscaleFactor); // neded for now as bindless not readly
        }
        else
        {
            allocate_usm(upscaleFactor);
        }

        return;
    }

    if(w == _w && h == _h)
        return;
    /* everything OK, nothing to do */

    if(w * h <= _max_w * _max_h)
    {
        if constexpr(sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>())
        {
            // Need to destroy and make again
            fprintf(stderr, "Need to deal with this realloc stufus\n");
            // Could all be done below
        }
        // NOTE: wrap the return in this if ^ with negation ! and it only runs when using usm
        else {}
        // TODO: Move to else when doing with texture meme version
        // smaller than current allocated segment hence it is fine to reuse in case of USM
        _w = w;
        _h = h;
        return;
    }

    // larger than current segment hence need to free and re-malloc

    sycl::free(_device_img, _device_queue);
    sycl::free(_device_src_img, _device_queue);

    _max_w = _w = w;
    _max_h = _h = h;

    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      _scaled_w * _scaled_h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);

    _device_img =
      popsift::sycl_common::malloc_devT<float>(_scaled_w * _scaled_h,
                                               __FILE__,
                                               __LINE__,
                                               "Could not allocate memory for float representation of image on device",
                                               _device_queue);
}

// Used for bindless image
void Image::allocate_bindless()
{
    //  Using bindless for input image (and scaling up with interpolation)

    fprintf(stderr, "Width and height used for bindless image w=%d h=%d", _w, _h);
    namespace syclexp = sycl::ext::oneapi::experimental;

    syclexp::image_descriptor img_desc(
      {static_cast<size_t>(_w), static_cast<size_t>(_h)}, 1, sycl::image_channel_type::unsigned_int8);

    // Normalized 0-1 indexing (for upscaling accessing inbetween pixels in horiz)
    // Uses linear interpolation (hardware accelerated it should be)
    syclexp::bindless_image_sampler img_sampler(
      sycl::addressing_mode::repeat, sycl::coordinate_normalization_mode::normalized, sycl::filtering_mode::linear);

    // Not using RAII version for clear destroy and alloc (usefull for resizing)
    _dev_img_handle = syclexp::alloc_image_mem(img_desc, _device_queue);

    // Uses wrapper function to supprt two overloads (due to documentation and install being different)
    _sampled_dev_img_handle =
      popsift::sycl_bindless::create_sampled_image(_dev_img_handle, img_sampler, img_desc, _device_queue);

    // Use this free function for
    // void free_image_mem(image_mem_handle memHandle,
    //                     image_type imageType,
    //                     const sycl::queue &syclQueue);

    //     enum class image_channel_type : /* unspecified */ {
    //   snorm_int8,
    //   snorm_int16,
    //   unorm_int8,
    //   unorm_int16,
    //   signed_int8,
    //   signed_int16,
    //   signed_int32,
    //   unsigned_int8,
    //   unsigned_int16,
    //   unsigned_int32,
    //   fp16,
    //   fp32,
    // };
    //
    // enum class image_type : /* unspecified */ {
    //   standard,
    //   mipmap,
    //   array,
    //   cubemap,
    // };
}

// Used for USM
void Image::allocate_usm(const float upscaleFactor)
{
    float scaleFactor = 1.0f / powf(2.0f, -upscaleFactor);

    _scaled_w = ceilf(_w * scaleFactor);
    _scaled_h = ceilf(_h * scaleFactor);

    _device_src_img = popsift::sycl_common::malloc_devT<unsigned char>(
      _scaled_w * _scaled_h, __FILE__, __LINE__, "Could not allocate memory for image on device", _device_queue);

    _device_img =
      popsift::sycl_common::malloc_devT<float>(_scaled_w * _scaled_h,
                                               __FILE__,
                                               __LINE__,
                                               "Could not allocate memory for float representation of image on device",
                                               _device_queue);
}

// directly making it normalized
sycl::event Image::load_divide(unsigned char* input)
{
    return _device_queue.submit([&](sycl::handler& cgh) {
        auto img = _device_img; // needed to avoid implicitly capturing this which is not allowed
        std::cout << "widht and height in load_divide" << _w << " - " << _h << std::endl;
        cgh.parallel_for(sycl::range<1>(_w * _h), [=](sycl::id<1> idx) {
            // To simulate normalized reads in PopSift -- think I would rather change the kernel in the future to
            // avoid this as I think that should be equivalent
            img[idx] = static_cast<float>(input[idx]) / 255.0f;
            // img[idx] = static_cast<float>(input[idx]);
        });
    });
}

// probably quite slow as it probably needs to copy over the input to device then
// do the manipulation and write to the disignated cuda_malloced area.
// but not sure if there is a better way (besides sampled_image and bindless_image)
// WARNING: CUDA's point (nearest neigbour) Seems to be a bit strange when it comes to
// choosing if it wants to take prev or next when position is perfectly inbetween texels like in popsift
// most of the time it takes prev like my implementation here but every now and then for a column it takes
// next and I'm not sure why it does that. Subtracting 0.000001 makes it take left all the time hovever it seems like
// but that makes the interpolation code wrong so can't be used in the cuda kernel.
// Must also be 0.000001 adding one more zero before the one makes the float to small and it goes back to choosing
// next in the odd cases
sycl::event Image::load_divide_point(unsigned char* input, const int& scaled_w)
{
    return _device_queue.submit([&](sycl::handler& cgh) {
        auto img = _device_img; // needed to avoid implicitly capturing this which is not allowed
        auto width = _w;
        int step = scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
        cgh.parallel_for(sycl::range<2>(_w, _h), [=](sycl::id<2> idx) {
            float pixel = static_cast<float>(input[idx[0] + idx[1] * width]) / 255.0;

            auto pos = idx[0] * step + idx[1] * step * scaled_w; // position in potentially upscaled image

            if(idx[0] == 5 && idx[1] == 5)
                sycl::ext::oneapi::experimental::printf("\n\n\t\t\tPos: %d -- step: %d", pos, step);

            // assumes contigious non padded memory -- which I believe is always the case
            // copy pixel to location and right and below and below to the right
            // just like a piint access from a texture would do.
            img[pos] = pixel;
            img[pos + 1] = pixel;
            img[pos + scaled_w] = pixel;
            img[pos + scaled_w + 1] = pixel;
        });
    });
}

// should mby look into storing image in local memory for this  but kernel will propbably not be used in final anyways
sycl::event Image::load_divide_linear(unsigned char* input, const int& scaled_w)
{
    return _device_queue.submit([&](sycl::handler& cgh) {
        auto img = _device_img; // needed to avoid implicitly capturing this which is not allowed
        auto width = _w;
        auto height = _h;
        int step = scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
        cgh.parallel_for(sycl::range<2>(_w, _h), [=](sycl::id<2> idx) {
            auto in_pos = idx[0] + idx[1] * width;

            auto in_pos_right = idx[0] == width - 1 ? in_pos : in_pos + 1;
            auto in_pos_down = idx[1] == height - 1 ? in_pos : in_pos + width;
            auto in_pos_down_right = (idx[0] == width - 1 && idx[1] == height - 1) ? in_pos
                                     : idx[0] == width - 1                         ? in_pos + width
                                     : idx[1] == height - 1                        ? in_pos + 1
                                                                                   : in_pos + width + 1; // default case

            float pixel = static_cast<float>(input[in_pos]) / 255.0;
            float pixel_right = static_cast<float>(input[in_pos_right]) / 255.0;
            float pixel_down = static_cast<float>(input[in_pos_down]) / 255.0;
            float pixel_down_right = static_cast<float>(input[in_pos_down_right]) / 255.0;

            auto pos = idx[0] * step + idx[1] * step * scaled_w; // position in potentially upscaled image

            img[pos] = pixel;
            img[pos + 1] = (pixel + pixel_right) / 2;
            img[pos + scaled_w] = (pixel + pixel_down) / 2;
            img[pos + scaled_w + 1] = (pixel + pixel_down + pixel_right + pixel_down_right) / 4;
        });
    });
}

// sycl::event Image::load_linear(unsigned char* input, const int& scaled_w)
// {
//     return _device_queue.submit([&](sycl::handler& cgh) {
//         auto img = _device_img; // needed to avoid implicitly capturing this which is not allowed
//         auto width = _w;
//         auto height = _h;
//         int step = scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
//         cgh.parallel_for(sycl::range<2>(width, height), [=](sycl::id<2> idx) {
//             auto in_pos = idx[0] + idx[1] * width;
//
//             auto in_pos_right = idx[0] == width - 1 ? in_pos : in_pos + 1;
//             auto in_pos_down = idx[1] == height - 1 ? in_pos : in_pos + width;
//             auto in_pos_down_right = (idx[0] == width - 1 && idx[1] == height - 1) ? in_pos
//                                      : idx[0] == width - 1                         ? in_pos + width
//                                      : idx[1] == height - 1                        ? in_pos + 1
//                                                                                    : in_pos + width + 1; // default
//                                                                                    case
//
//             float pixel = static_cast<float>(input[in_pos]);
//             float pixel_right = static_cast<float>(input[in_pos_right]);
//             float pixel_down = static_cast<float>(input[in_pos_down]);
//             float pixel_down_right = static_cast<float>(input[in_pos_down_right]);
//
//             auto pos = idx[0] * step + idx[1] * step * scaled_w; // position in potentially upscaled image
//
//             img[pos] = pixel;
//             img[pos + 1] = (pixel + pixel_right) / 2;
//             img[pos + scaled_w] = (pixel + pixel_down) / 2;
//             img[pos + scaled_w + 1] = (pixel + pixel_down + pixel_right + pixel_down_right) / 4;
//         });
//     });
// }

// Only valid of the load functions others pass a host pointer to use which don't work on gpu need to transfer the image
// to device first before kernel launch
sycl::event Image::load_linear(sycl::event src_img_transfer)
{
    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(src_img_transfer);
        auto img = _device_img;
        auto input = _device_src_img;
        auto width = _w;
        auto height = _h;
        int step = _scaled_w / width; // floored -- not sure if it is corretc for other than 1 and 2
        int scaled_w = _scaled_w;
        cgh.parallel_for(sycl::range<2>(width, height), [=](sycl::id<2> idx) {
            auto in_pos = idx[0] + idx[1] * width;

            auto in_pos_right = idx[0] == width - 1 ? in_pos : in_pos + 1;
            auto in_pos_down = idx[1] == height - 1 ? in_pos : in_pos + width;
            auto in_pos_down_right = (idx[0] == width - 1 && idx[1] == height - 1) ? in_pos
                                     : idx[0] == width - 1                         ? in_pos + width
                                     : idx[1] == height - 1                        ? in_pos + 1
                                                                                   : in_pos + width + 1; // default case

            float pixel = static_cast<float>(input[in_pos]);
            float pixel_right = static_cast<float>(input[in_pos_right]);
            float pixel_down = static_cast<float>(input[in_pos_down]);
            float pixel_down_right = static_cast<float>(input[in_pos_down_right]);

            auto pos = idx[0] * step + idx[1] * step * scaled_w; // position in potentially upscaled image

            img[pos] = pixel;
            img[pos + 1] = (pixel + pixel_right) / 2;
            img[pos + scaled_w] = (pixel + pixel_down) / 2;
            img[pos + scaled_w + 1] = (pixel + pixel_down + pixel_right + pixel_down_right) / 4;
        });
    });
}

// // only for printing and debugging
// sycl::event Image::host_move(void* output) { return _device_queue.memcpy(output, _device_img, _w * _h); };
//
// // only for printing and debugging -- quite inefficient
// void Image::print_region(int start_x, int start_y, int end_x, int end_y)
// {
//     using std::cout;
//     using std::endl;
//     using std::printf;
//     cout << "Inside of Print_region of Image" << endl << endl << endl;
//
//     unsigned char* img = (unsigned char*)malloc(_w * _h * sizeof(unsigned char));
//     if(img == NULL)
//     {
//         cout << "Memory allocation failed" << endl;
//         return;
//     }
//     sycl::event write_event = host_move(img);
//
//     if(start_x > _w || end_x > _w || start_y > _h || end_y > _h)
//     {
//         cout << "Region coordinates are outisde of bounds of Image" << endl;
//         return;
//     }
//     if(start_x > end_x || start_y > end_y)
//     {
//         cout << "Invalid region" << endl;
//         return;
//     }
//     if(start_x < 0 || start_y < 0 || end_x < 0 || end_y < 0)
//     {
//         cout << "Cannot have negative position of region" << endl;
//         return;
//     }
//     printf("Image region (%d, %d) -> (%d, %d)\n", start_x, start_y, end_x, end_y);
//
//     try
//     {
//         write_event.wait();
//     }
//     catch(const sycl::exception& e)
//     {
//         std::cerr << "SYCL exception caught: " << e.what() << std::endl;
//         return;
//     }
//     // _device_queue.wait(); // wait for memcpy to finish
//     for(int i = start_y; i < end_y; ++i)
//     {
//         for(int j = start_x; j < end_x; ++j)
//         {
//             printf("%03u ", img[i * _w + j]);
//         }
//         cout << endl;
//     }
//     cout << endl << endl;
//     free(img);
// }

} // namespace popsift
